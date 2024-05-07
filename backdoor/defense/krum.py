import torch
import numpy as np

from base import Defense, vectorize_net, vectorize_dict
from utils import logger

class Krum(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """
    def __init__(self, mode, num_adv, *args, **kwargs):
        assert (mode in ("krum", "multi-krum"))
        self._mode = mode
        self.num_workers = None # number of selected clients
        self.s = num_adv # number of poison clients supposed in the selected clients

    def exec(self, client_models, clients_this_round, num_dps, device, key_order, *args, **kwargs):
        
        self.num_workers = len(clients_this_round)

        vectorize_nets = [vectorize_dict(client_models[client_idx], key_order).detach() for client_idx in  clients_this_round]

        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(torch.norm(g_i - g_j).pow(2).item())
            neighbor_distances.append(distance)

        
        # compute scores
        nb_in_score = self.num_workers - self.s - 2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])
            # alternative to topk in PyTorch
            dists_tensor = torch.tensor(dists)
            topk_values, _ = torch.topk(dists_tensor, nb_in_score)
            scores.append(torch.sum(topk_values).item())
            
        if self._mode == "krum":
            i_star = scores.index(min(scores))

            n_freq = torch.zeros(len(vectorize_nets), device=device)
            n_freq[i_star] = 1.0
            
            logger.info(f"@@@@ Krum select {clients_this_round[i_star]} client")

            # logger.info("@@@@ The chosen one is user: {}, which is global user: {}".format(scores.index(min(scores)), g_user_indices[scores.index(min(scores))]))
            # aggregated_model = client_models[0]  # create a clone of the model
            # aggregated_model.load_state_dict(vectorize_nets[i_star].to(device))
            # neo_net_list = [aggregated_model]
            # logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            # neo_net_freq = [1.0]
            # return neo_net_list, neo_net_freq
            return n_freq

        elif self._mode == "multi-krum":
            topk_ind = np.argpartition(scores, nb_in_score+2)[:nb_in_score+2]

            # We reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]
            reconstructed_freq = torch.tensor([snd/sum(selected_num_dps) for snd in selected_num_dps], dtype=torch.float32, device=device)
            
            logger.info(f"@@@@ Multi-Krum select {[clients_this_round[i] for i in topk_ind]} client")
            return reconstructed_freq

            logger.info("Num data points: {}".format(num_dps))
            logger.info("Num selected data points: {}".format(selected_num_dps))
            logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind, [g_user_indices[ti] for ti in topk_ind]))
            
            aggregated_grad = torch.sum(torch.stack([reconstructed_freq[i] * vectorize_nets[j] for i, j in enumerate(topk_ind)], dim=0), dim=0)  # Weighted sum of the gradients
            
            aggregated_model = client_models[0]  # create a clone of the model
            load_model_weight(aggregated_model, aggregated_grad)
            neo_net_list = [aggregated_model]
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq