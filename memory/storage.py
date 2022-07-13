
from math import*
from decimal import Decimal
from torch import nn
import torch.nn.functional as F
class Similarity():
 
    """ Five similarity measures function """
 
    def euclidean_distance(self,x,y):
 
        """ return euclidean distance between two lists """
 
        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
 
    def manhattan_distance(self,x,y):
 
        """ return manhattan distance between two lists """
 
        return sum(abs(a-b) for a,b in zip(x,y))
 
    def minkowski_distance(self,x,y,p_value):
 
        """ return minkowski distance between two lists """
 
        return self.nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),
           p_value)
 
    def nth_root(self,value, n_root):
 
        """ returns the n_root of an value """
 
        root_value = 1/float(n_root)
        return round (Decimal(value) ** Decimal(root_value),3)
 
    def cosine_similarity(self,x,y):
        """ return cosine similarity between two lists """
        print("cosine_similaritycosine_similaritycosine_similarity")
        numerator = sum(a*b for a,b in zip(x,y))
        denominator = self.square_rooted(x)*self.square_rooted(y)
        return numerator/float(denominator)
        # return round(numerator/float(denominator),3)
 
    def square_rooted(self,x):
 
        """ return 3 rounded square rooted value """
 
        return round(sqrt(sum([a*a for a in x])),3)
 
    def jaccard_similarity(self,x,y):
 
        """ returns the jaccard similarity between two lists """
 
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)


import numpy as np
import torch
import copy

class Storage():
    def __init__(self,bs,sim_type):
        # self.num_storage = bs//2
        self.num_storage = 15 #temp

        self.list_storage = []
        self.scalar_storage=dict()
        self.img_storage = dict()
        self.func_similar = Similarity()
        self.sim_type = sim_type

    def get_similarity(self,feat1,feat2):
        # print(feat1.shape, feat2.shape)
        # print("get_similarityget_similarityget_similarity")
        feat1 = torch.flatten(feat1)
        feat2 = torch.flatten(feat2)
        if self.sim_type == 'cosign':
            val = self.func_similar.cosine_similarity(feat1, feat2)
        elif self.sim_type == 'euclid':
            val = self.func_similar.euclidean_distance(feat1,feat2)
        return val

    
    def pop_feature(self, feature):
        # print("get_mixed_featureget_mixed_featureget_mixed_featureget_mixed_feature")
        _feature = feature
        if self.list_storage:
            popped_storage = self.list_storage.pop()
            _feature = 0.5*_feature + 0.5*popped_storage
        return _feature
    
    def pop_features(self, features):
            # print("get_mixed_featureget_mixed_featureget_mixed_featureget_mixed_feature")
        if self.list_storage:
            popped_storage = self.list_storage.pop()
            #temp
            feature_t = features[0]
            feature_s = features[1]
#             feature_t = 0.5*features[0].cpu().detach() + 0.5*popped_storage[0]
#             feature_s = 0.5*features[1].cpu().detach() + 0.5*popped_storage[1]
            
            features = [feature_t, feature_s]
            # print(features)
        return features

    def get_img(self, path):
        if path in self.img_storage :
            return path, self.img_storage[path]
        else : 
            print('NO IMAGE STORAGE !')
            return (None, None)
        
    def get_mixed_feature(self, feature):
        # print("get_mixed_featureget_mixed_featureget_mixed_featureget_mixed_feature")
        _feature = feature
        # _feature = copy.deepcopy(feature.cpu().detach())
        if self.sim_type == 'cosign':
            val_max, idx_max = self.get_maximum_similar(copy.deepcopy(feature.cpu().detach()).cuda())
        elif self.sim_type =='euclid':
            val_max, idx_max = self.get_minimum_similar(feature.cpu().detach()).cuda()
            # val_max, idx_max = self.get_minimum_similar(copy.deepcopy(feature.cpu().detach()).cuda())

        if idx_max >= 0:
            _feature = 0.5*_feature + 0.5*self.list_storage[idx_max]
        return _feature
            
    def get_maximum_similar(self, tar_feat):
        val_max = float('-inf')
        idx_max = -1
        for idx, feat in enumerate(self.list_storage):
            # print(tar_feat)
            _val = self.get_similarity(feat, tar_feat)
            if val_max < _val :
                val_max = _val
                idx_max = idx
        return val_max, idx_max

    def get_minimum_similar(self, tar_feat):
        val_min = float('inf')
        idx_min = -1
        for idx, feat in enumerate(self.list_storage):
            _val = self.get_similarity(feat, tar_feat)
            if val_min > _val :
                val_min = _val
                idx_min = idx
        return val_min, idx_min

    def update_storage(self, feature, mode='minimum'):
        if mode == 'minimum':
            if len(self.list_storage) == self.num_storage:
                val_min, idx_min = self.get_minimum_similar(copy.deepcopy(feature.cpu().detach()).cuda())
                self.list_storage[idx_min] = feature
            else:
                self.list_storage.append(feature)
        elif mode == 'maximum':
            if len(self.list_storage) == self.num_storage:
                val_max, idx_max = self.get_maximum_similar(copy.deepcopy(feature.cpu().detach()).cuda())
                self.list_storage[idx_max] = feature
            else:
                self.list_storage.append(feature)
                
    def update_storage_by_representation_redesign(self, path,feature, mode = 'minimum'):
        #Maximum : 가장 먼 feature를 가져옴
        remove_key, key, val = None, None, None
        scalar = torch.mean(feature)
        if mode == 'minimum':
            remove_key, _ = self.get_maximum_scalar(copy.deepcopy(feature.cpu().detach()).cuda())
            key, val = self.get_minimum_scalar(copy.deepcopy(feature.cpu().detach()).cuda())
        else: #maximum
            remove_key, _ = self.get_minimum_scalar(copy.deepcopy(feature.cpu().detach()).cuda())
            key, val = self.get_maximum_scalar(copy.deepcopy(feature.cpu().detach()).cuda())
                
        if len(self.list_storage) == self.num_storage:
            self.img_storage.pop(remove_key)
            self.scalar_storage.pop(remove_key)

#         print(f'removed key is {remove_key} | key and val are {key}, {val}')
        self.img_storage[path] = feature
        self.scalar_storage[path] = scalar

                
#     def update_storage_by_representation(self, path,feature):
#         scalar = torch.mean(feature)
#         if len(self.scalar_storage) == self.num_storage:
#             self.scalar_storage.popitem()
#         self.scalar_storage[path] = scalar
#         self.img_storage[path] = feature

    def get_minimum_scalar(self, tar_feat):
        if not self.scalar_storage:
            print('NOT IN THE SCALAR IN STORAGE !!!')
            return (None,None)
        
        scalar = torch.mean(tar_feat)
        res_key, res_val = min(self.scalar_storage.items(), key=lambda x: abs(scalar - x[1]))
        return (res_key, res_val)
    
    def get_maximum_scalar(self, tar_feat):
        if not self.scalar_storage:
            print('NOT IN THE SCALAR IN STORAGE !!!')
            return (None,None)
        
        scalar = torch.mean(tar_feat)
        res_key, res_val = max(self.scalar_storage.items(), key=lambda x: abs(scalar - x[1]))
        return (res_key, res_val)
    
    def get_softmax_distance(self, list_scalar):
        list_softmax = F.softmax(torch.tensor(list_scalar).float(), dim=0)
        dist = abs(list_softmax[0]-list_softmax[1])
        return dist
    
    def get_mixed_img(self, img_s, img_t, alpha=0.2):
        return img_s*(1.-alpha)+img_t*(alpha)
            