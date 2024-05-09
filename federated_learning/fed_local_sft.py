import torch
import copy
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from trl import SFTTrainer
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from utils import logger

def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary, global_auxiliary, is_poison_client=False, backdoor_train_args=None, key_order=None, overall_temp=None, eos_token=None, neurotoxin_ratio=None, device=None):
    
    if fed_args.fed_alg == 'fedprox':
        trainer = SFTTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = SFTTrainerSCAFFOLD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
        )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    elif (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    
    if is_poison_client: 
        if backdoor_train_args.mode == "pgd":
            #1. obtain trianable params
            #2. project peft param
            pgd_callback = PGD_Callback(global_dict,
                                        backdoor_train_args.pgd_eps, 
                                        backdoor_train_args.pgd_gamma, 
                                        backdoor_train_args.pgd_project_freq, 
                                        key_order=key_order)
            trainer.add_callback(pgd_callback)
            
        elif backdoor_train_args.mode == "neurotoxin":
            grad_mask_dict = get_grad_mask(model, tokenizer, local_dataset, overall_temp, 16, eos_token, device, neurotoxin_ratio)
            neurotoxin_callback = Neurotoxin_Callback(grad_mask_dict)
            trainer.add_callback(neurotoxin_callback)

        elif backdoor_train_args.mode == "native":
            pass
        else:
            raise ValueError(f'Unsupported `mode`: {backdoor_train_args.mode}')
    
    return trainer

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para

class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)


class PGD_Callback(TrainerCallback):
    def __init__(self, global_state, pgd_eps, pgd_gamma, pgd_project_freq, key_order) -> None:
        super().__init__()

        self.global_state = global_state
        self.pgd_eps = pgd_eps
        self.pgd_gamma = pgd_gamma
        self.pgd_project_freq = pgd_project_freq
        self.key_order = key_order
        
        self.global_vector = self.dict_param_to_vector(self.global_state)
        
    def dict_param_to_vector(self, param_dict):
        return torch.cat([param_dict[key].flatten() for key in self.key_order])
    
    def vector_to_dict_param(self, vector, param_dict):
        start = 0
        max_num = vector.shape[0]
        new_params = {}
        
        for key in self.key_order:
            end = start + param_dict[key].numel()
            if end > max_num:
                raise ValueError(f"vector size {max_num} is not enough for {end} params")

            new_params[key] = vector[start:end].view_as(param_dict[key])
            start = end
        return new_params
    
    def on_step_end(self, args, state, control, model,  **kwargs):
        
        if state.global_step % self.pgd_project_freq == 0 or state.global_step == state.max_steps:

            local_param = get_peft_model_state_dict(model)
            local_vector = self.dict_param_to_vector(local_param)
            # global_vector = self.dict_param_to_vector(global_param)
            
            diff_vector = local_vector - self.global_vector
            
            local_proj_vector = self.global_vector + self.pgd_eps * diff_vector / torch.norm(diff_vector)

            local_proj_param = self.vector_to_dict_param(local_proj_vector, local_param)
            
            set_peft_model_state_dict(model, local_proj_param)
            
class Neurotoxin_Callback(TrainerCallback):
    def __init__(self, grad_mask_dict):
        self.grad_mask_list = grad_mask_dict
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, optimizer, **kwargs):
        ori_optimizer_step = optimizer.step
        def masked_step(closure=None):
            for name, param in model.named_parameters():
                if name in self.grad_mask_list.keys():
                    if not param.requires_grad:
                        raise ValueError("")
                    elif param.grad is not None:
                        param.grad *= self.grad_mask_list[name]
                        
            ori_optimizer_step(closure)
        optimizer.step = masked_step
                    
        

#TODO:Test mask func
def get_grad_mask(model, tokenizer, dataset, overall_temp, batch_size, eos_token, device=None, ratio=None):
    
    logger.info(f"Getting grad mask for {model.__class__.__name__}")
    model.train()
    model.zero_grad()
    
    #use not poison instruction and response
    inputs_str = [overall_temp.format(ex["instruction"], ex["response"], eos_token) for ex in dataset]
    from tqdm import tqdm
    
    for i in tqdm(range(0, len(inputs_str), batch_size)):
        batch_inputs = inputs_str[i:i+batch_size]
        
        inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs["labels"] = inputs["input_ids"].clone()
        inputs = {key: val.to(device) for key, val in inputs.items()}

        loss = model(**inputs).loss
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    mask_grad_dict = {}

    for name, parms in model.named_parameters():
        if parms.requires_grad:
            gradients = parms.grad.abs().view(-1)
            gradients_length = len(gradients)
            _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1.0
            mask_grad_dict[name] = mask_flat.reshape(parms.grad.size()).to(device)

    model.zero_grad()
    return mask_grad_dict