import random
import torch

def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients <= fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        # if len(total_clients_idxs) < fed_args.sample_clients:
            # clients_this_round = total_clients_idxs
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round


def get_clients_this_round_with_poison(fed_args, round, clean_clients_idxs, poison_clients_idxs, attack_args):
    if attack_args.attack_mode == "random" or not attack_args.poison.use_poison:
        
        # if attack_window[1] <= 1:
        #    attack_rounds_start, attack_rounds_end = int(fed_args.num_rounds * attack_window[0]), int(fed_args.num_rounds * attack_window[1])
        # else:
        #    attack_rounds_start, attack_rounds_end = int(attack_window[0]), int(attack_window[1])
        
        # # breakpoint()
        # if round >= attack_rounds_start and round <= attack_rounds_end:
        #     return get_clients_this_round(fed_args, round, poison_clients_idxs+clean_clients_idxs)
        # else:
        #     return get_clients_this_round(fed_args, round, clean_clients_idxs)
              
        return get_clients_this_round(fed_args, round)
    elif attack_args.poison_mode == "fix-frequency":
        random.seed(round)

        clean_samples = fed_args.num_clients if fed_args.num_clients < fed_args.sample_clients else fed_args.sample_clients
        if (round - attack_args.poison.start_round) % attack_args.poison.interval == 0:
            poison_client = random.choice(poison_clients_idxs)
            clean_samples -= 1

        clean_clients = random.sample(clean_clients_idxs, clean_samples)
        
        return clean_clients + [poison_client]
        
    elif (attack_args.attack_mode).startswith("local"):
        client_id = int((attack_args.attack_mode).split("_")[-1])
        return [client_id]
    
    elif (attack_args.attack_mode).startswith("total"):
        client_num = int((attack_args.attack_mode).split("_")[-1])
        return list(range(client_num))
        
    else:
        raise ValueError(f"Unsupported poison mode: {attack_args.poison_mode}")
            



def global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, n_freq=None, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None):
    
    if n_freq is None:
        sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
        n_freq = [sample_num_list[client] / sample_this_round for client in clients_this_round]

    global_auxiliary = None

    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
    
    elif fed_args.fed_alg == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedavg':
        ## TODO: check the correctness of inserted n_freq
        delta_w = {}
        for key in global_dict.keys():
            delta_w[key] = sum([(local_dict_list[client][key] - global_dict[key]) * n_freq[i] for i, client in enumerate(clients_this_round)])
            global_dict[key] +=  delta_w[key]
            
    else:   # Normal dataset-size-based aggregation 
        for key in global_dict.keys():
            try:
                #BUG: quantization model would add xxx.SCB and xxx.xxx_weight_format key to state_dict, their value are str, it can execute the following computation
                # First dequantization：w_d = (w_qantize * SCB) / 127
                # Merge: w_AB = w_dA * w_A + w_dB * w_B = (w_A * w_qantize_A * SCB_A + w_B * w_qantize_A * SCB_B) / 127
                # quantize and calculate w_quantize_AB and SCB_AB
                
                global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
            except:
                breakpoint()
    
    return global_dict, global_auxiliary