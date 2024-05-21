
import torch
import numpy as np

from .base import Defense, vectorize_net, vectorize_dict
from utils import logger

import sklearn.metrics.pairwise as smp


class FoolsGold(Defense):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clients = None
        self.n_features = None
        self.n_classes = None

    def get_cos_similarity(self, full_deltas):
        '''
        Returns the pairwise cosine similarity of client gradients
        '''
        if True in np.isnan(full_deltas):
            pdb.set_trace()
        return smp.cosine_similarity(full_deltas)

    def importanceFeatureMapGlobal(self, model):
        # aggregate = np.abs(np.sum( np.reshape(model, (10, 784)), axis=0))
        # aggregate = aggregate / np.linalg.norm(aggregate)
        # return np.repeat(aggregate, 10)
        return np.abs(model) / np.sum(np.abs(model))

    def importanceFeatureMapLocal(self, model, topk_prop=0.5):
        # model: np arr
        d = self.n_features # dim of flatten weight
        class_d = int(d / self.n_classes)

        M = model.copy()
        M = np.reshape(M, (self.n_classes, class_d))
        
        # #Take abs?
        # M = np.abs(M)

        for i in range(self.n_classes):
            if (M[i].sum() == 0):
                pdb.set_trace()
            M[i] = np.abs(M[i] - M[i].mean())
            
            M[i] = M[i] / M[i].sum()

            # Top k of 784
            topk = int(class_d * topk_prop)
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]
            M[i][sig_features_idx] = 0
        
        return M.flatten()   

    def importanceFeatureHard(self, model, topk_prop=0.5):

        class_d = int(self.n_features / self.n_classes)

        M = np.reshape(model, (self.n_classes, class_d))
        importantFeatures = np.ones((self.n_classes, class_d))
        # Top k of 784
        topk = int(class_d * topk_prop)
        for i in range(self.n_classes):
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]     
            importantFeatures[i][sig_features_idx] = 0
        return importantFeatures.flatten()  

    def get_krum_scores(self, X, groupsize):

        krum_scores = np.zeros(len(X))

        # Calculate distances
        distances = np.sum(X**2, axis=1)[:, None] + np.sum(
            X**2, axis=1)[None] - 2 * np.dot(X, X.T)

        for i in range(len(X)):
            krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])

        return krum_scores

    def foolsgold(self, this_delta, summed_deltas, sig_features_idx, iter, model, topk_prop=0, importance=False, importanceHard=False, clip=0):
        epsilon = 1e-5
        # Take all the features of sig_features_idx for each clients
        sd = summed_deltas.copy()
        sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)

        if importance or importanceHard:
            if importance:
                # smooth version of importance features
                importantFeatures = self.importanceFeatureMapLocal(model, topk_prop)
            if importanceHard:
                # hard version of important features
                importantFeatures = self.importanceFeatureHard(model, topk_prop)
            for i in range(self.n_clients):
                sig_filtered_deltas[i] = np.multiply(sig_filtered_deltas[i], importantFeatures)
                
        N, _ = sig_filtered_deltas.shape
        

        
        # cs = np.zeros((N,N))
        # for i in range(N):
        #     for j in range(N):
        #         if i == j:
        #             cs[i,i] = 1  
        #             continue
        #         if cs[i,j] != 0 and cs[j,i] != 0:
        #             continue
        #         dot_i = sig_filtered_deltas[i][np.newaxis, :] @ sig_filtered_deltas[j][:, np.newaxis]
        #         norm_mul = np.linalg.norm(sig_filtered_deltas[i]) * np.linalg.norm(sig_filtered_deltas[j])
        #         cs[i, j] = cs[j, i] = dot_i / (norm_mul + epsilon)
        # cs = cs - np.eye(N)
        
        cs = smp.cosine_similarity(sig_filtered_deltas) - np.eye(N)
        # Pardoning: reweight by the max value seen
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(self.n_clients):
            for j in range(self.n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)

        wv[(wv == 1)] = .99
        
        # Logit function
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        if clip != 0:

            # Augment onto krum
            scores = self.get_krum_scores(this_delta, self.n_clients - clip)
            bad_idx = np.argpartition(scores, self.n_clients - clip)[(self.n_clients - clip):self.n_clients]

            # Filter out the highest krum scores
            wv[bad_idx] = 0

        print(f"wv: {wv}")
        wv = wv/sum(wv)
        # avg_updates = np.average(this_delta, axis=0, weights=wv)
        # return avg_updates, wv
        return  wv

    def exec(self, delta, summed_deltas, net_avg, r,  device, num_clients, num_features, key_order, *args, **kwargs):
        '''
        Aggregates history of gradient directions
        '''

        self.n_clients = num_clients
        self.n_features = num_features
        
        print(f"START Aggregating history of gradient directions")
        vectorize_avg_net = vectorize_dict(net_avg,key_order).detach().cpu().numpy()
        # flatten_net_avg = vectorize_dict(net_avg, key_order).detach().cpu().numpy()

        # Significant features filter, the top k biggest weights
        # topk = int(self.n_features / 2)
        # sig_features_idx = np.argpartition(flatten_net_avg, -topk)[-topk:]
        sig_features_idx = np.arange(self.n_features)
        # avg_delta, wv = self.foolsgold(delta, summed_deltas, sig_features_idx, r, vectorize_avg_net, clip = 0)
        wv = self.foolsgold(delta, summed_deltas, sig_features_idx, r, vectorize_avg_net, clip = 0)
        return wv