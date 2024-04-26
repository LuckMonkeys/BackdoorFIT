from peft import get_peft_model_state_dict, set_peft_model_state_dict
import torch

def get_model_state(model, is_peft=False):
    if is_peft:
        return get_peft_model_state_dict(model)
    else:
        return model.state_dict()

def set_model_state(model, state_dict, is_peft=False):
    if is_peft:
        set_peft_model_state_dict(model, state_dict)
    else:
        if torch.cuda.is_available():
            model.cuda()
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError("CUDA is not available. Please check your installation.")

    