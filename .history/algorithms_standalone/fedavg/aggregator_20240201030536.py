import copy
from http import client
import logging
import random
import time

import numpy as np
import torch
from .client import FedAVGClientTimer
import decimal

from utils.data_utils import (
    get_data,
    apply_gradient,
    average_named_params,
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations
)

from algorithms_standalone.basePS.aggregator import Aggregator


from model.build import create_model
from experiments.cca_compare.cca_compare import two_same_models_compare
# from utils.corr_methods import CCA
from sklearn.cross_decomposition import CCA
from utils.svcca import svcca
from sklearn.decomposition import PCA, KernelPCA
from utils.corr_methods_CKA import cka, gram_linear, gram_rbf
from utils.mmd_loss import MMDLoss as MMD

class FedAVGAggregator(Aggregator):
    def __init__(self, train_global, test_global, all_train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                args, model_trainer, perf_timer=None, metrics=None, traindata_cls_counts=None):
        super().__init__(train_global, test_global, all_train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                args, model_trainer, perf_timer, metrics, traindata_cls_counts=traindata_cls_counts)
        self.model_list=[]
        self.dist_list=[]
        if self.args.scaffold:
            self.c_model_global = create_model(self.args,
                model_name=self.args.model, output_dim=self.args.model_output_dim)
            for name, params in self.c_model_global.named_parameters():
                params.data = params.data*0

    def get_max_comm_round(self):
        # if self.args.HPFL:
            # return 1
        return self.args.comm_round
        # return self.args.max_epochs // self.args.global_epochs_per_round + 1
        # return 1
    
    def test_CCA(self, test_data, select_index):
        CCA_list=self.cal_CCA(self.dist_list, test_data, select_index)
        return np.array(CCA_list), np.argmax(CCA_list)
        
    def select_pluggable(self, test_data, select_index, client_list=None):
        # print(self.dist_list)
        # print(self.test_data_local_dict[client_index])
        if select_index=='KL':
            KL_list=self.cal_KL(self.dist_list, test_data)# get the data of corresponding client(feature of test_data should uploaded by client itself)
            
            # print("np.argmin(KL_list)====================",np.argmin(KL_list))
            # print("KL_list==============",KL_list)
            return np.argmin(KL_list)
        elif select_index=='distance':
            distance_array, feat=self.cal_distance(self.dist_list, test_data)
            # print(distance_array.shape)
            return np.argmin(distance_array, axis=1), feat
        elif select_index=="CCA":
            CCA_list=self.cal_CCA(self.dist_list, test_data, select_index)
            score = np.array(CCA_list)
            choice = np.argmax(CCA_list)
        elif "CKA" in select_index:
            CKA_list=self.cal_CCA(self.dist_list, test_data, select_index)
            score = np.array(CKA_list)
            choice = np.argmax(CKA_list)
        elif select_index=="multi-Pluggable_with_CCA":
            CCA_list=self.cal_CCA(self.dist_list, test_data)
            # default for argsort: ascending index
            return np.argsort(CCA_list)[::-1][:self.args.num_Pluggable]
        elif select_index=="multi-Pluggable_with_mean":
            distance_array, _=self.cal_distance(self.dist_list, test_data)
            distance_list=np.argmin(distance_array, axis=1)
            return distance_list[::-1][:self.args.num_Pluggable]
        elif select_index=="index":
            return np.array(range(len(self.dist_list))).astype(int)
        elif select_index=="multi-Pluggable_with_OOD":
            OOD_list=self.cal_OOD(client_list, test_data)
            return np.argsort(OOD_list)[::-1][:self.args.num_Pluggable]
        elif select_index=="personalized_with_OOD":
            OOD_list=self.cal_OOD(client_list, test_data)
            score = np.array(OOD_list)
            choice = np.argmax(OOD_list)
        elif select_index=="OOD":
            OOD_array, feat=self.cal_OOD_per_sample(client_list, test_data)
            return np.argmax(OOD_array, axis=1),feat
        elif select_index=="MMD":
            MMD_list=self.cal_MMD(self.dist_list, test_data)
            score = np.array(MMD_list)
            choice = np.argmin(MMD_list)
    
    def cal_KL(self,pluggable_dist,test_loader):
        # extract distribution from test_loader
        feature_extractor=torch.nn.Sequential(*list(self.trainer.model.children())[:-self.args.freeze_layer]).to(self.device)
        feature=[]
        if self.args.if_eval:
            feature_extractor.eval()
        with torch.no_grad():
            for _,(X,_) in enumerate(test_loader):
                X=X.to(self.device)
                if self.args.freeze_layer < len(list(self.trainer.model.children())):
                    feature.append(feature_extractor(X).cpu().detach().numpy())# train_data include X and y
                else:
                    feature.append(X.cpu().detach().numpy())
        feature=np.concatenate(feature, axis=0)
        feature=np.squeeze(feature).reshape(-1, np.prod(feature.shape[1:]) )
        mean_test=np.mean(feature,axis=0).flatten() # suppose the sampling dimension is 0
        var_test=1/feature.shape[0]*((feature-mean_test).T@(feature-mean_test))
        mean_test=np.expand_dims(mean_test,axis=-1)
        mean_plug=list(zip(*pluggable_dist))[0]
        var_plug=list(zip(*pluggable_dist))[1]
        # calculate KL divengence as the feature is normally distributed 
        KL_list=[]
        
        for i in range(len(mean_plug)):
            if np.linalg.det(var_test.astype(np.float64))==0.0 or np.linalg.det(var_plug[i].astype(np.float64))==0.0:
                raise RuntimeError
            KL=1/2*(np.log(np.linalg.det(var_test.astype(np.float64))/np.linalg.det(var_plug[i].astype(np.float64)))+np.trace(np.linalg.inv(var_test)@var_plug[i])+np.squeeze((mean_plug[i]-mean_test).T@np.linalg.inv(var_test)@(mean_plug[i]-mean_test))-mean_test.shape[0])
            KL_list.append(KL)
        return KL_list
    
    def cal_distance(self, pluggable_dist, test_loader):
        feature_extractor=torch.nn.Sequential(*list(self.trainer.model.children())[:-self.args.freeze_layer]).to(self.device)
        feature=[]
        if self.args.if_eval:
            feature_extractor.eval()
        with torch.no_grad():
            for _,(X,_) in enumerate(test_loader):
                X=X.to(self.device)
                if self.args.freeze_layer < len(list(self.trainer.model.children())):
                    feature.append(feature_extractor(X).cpu().detach().numpy())# train_data include X and y
                else:
                    feature.append(X.cpu().detach().numpy())
        feature=np.concatenate(feature, axis=0)
        print(f"when layer={self.args.freeze_layer}, feature shape======",feature.shape)
        feat=np.squeeze(feature)
        feature=np.squeeze(feature).reshape(-1, np.prod(feature.shape[1:]))
        if self.args.feature_reduction_dim>0:
            if self.args.freeze_layer==1:
                feature=feature.reshape(-1, 1, np.prod(feature.shape[1:]))
                feature_reductor=torch.nn.Sequential(
                    torch.nn.MaxPool1d((feature.shape[-1]//self.args.feature_reduction_dim))
                )
                feature=feature_reductor(torch.Tensor(feature)).detach().numpy()
                feature=np.squeeze(feature)
            elif self.args.freeze_layer==3:
                feature=feat.reshape(-1, )

        distance_list=[]
        # print(pluggable_dist[0][0].shape)
        # print(pluggable_dist[0][1])
        assert pluggable_dist!=[]
        
        for plug in pluggable_dist:
            mean_plug=plug[0]
            if self.args.freeze_layer==1:
                if self.args.feature_reduction_dim>0:
                    mean_plug=mean_plug.reshape(1, 1, -1)
                    mean_plug=feature_reductor(torch.Tensor(mean_plug)).detach().numpy()
                    mean_plug=np.squeeze(mean_plug, axis=0).T
            distance_list.append(np.sum((feature-mean_plug.T)**2, axis=1, keepdims=False))
        return np.array(distance_list).T, feat

    def cal_OOD_per_sample(self, client_list, test_loader):
        feature_extractor=torch.nn.Sequential(*list(self.trainer.model.children())[:-self.args.freeze_layer]).to(self.device)
        feature=[]
        if self.args.if_eval:
            feature_extractor.eval()
        with torch.no_grad():
            for _,(X,_) in enumerate(test_loader):
                X=X.to(self.device)
                if self.args.freeze_layer < len(list(self.trainer.model.children())):
                    feature.append(feature_extractor(X).cpu().detach().numpy())# train_data include X and y
                else:
                    feature.append(X.cpu().detach().numpy())
        feature=np.concatenate(feature, axis=0)
        print(f"when layer={self.args.freeze_layer}, feature shape======",feature.shape)
        feat=np.squeeze(feature)
        logits_list=[]
        # get logits from all plug
        for idx,client in enumerate(client_list):
            if self.args.OOD_independ_classifier:
                model=client.trainer.OOD_dete.to(self.device)
            else:
                model=client.trainer.model.to(self.device)
            if self.args.if_eval:
                model.eval()
            logit=[]
            with torch.no_grad():
                for batch_idx, (x, labels) in enumerate(test_loader):
                    x = x.to(self.device)
                    labels = labels.to(self.device)
                    real_batch_size = labels.shape[0]
                    if self.args.model_input_channels == 3 and x.shape[1] == 1:
                        x = x.repeat(1, 3, 1, 1)
                    if self.args.model_out_feature:
                        output, feat = model(x)
                    else:
                        output = model(x)
                    logit.append(output.cpu().detach())
            logits=torch.concat(logit,axis=0)
            logits=torch.squeeze(logits)
            logits_list.append(logits)
        if self.args.OOD_independ_classifier:
            OOD_array=torch.softmax(torch.stack(logits_list),dim=-1)
            res=OOD_array[...,1].numpy().T
        else:
            OOD_array=np.array(logits_list)[...,-1]
            res=OOD_array.T
        np.set_printoptions(threshold=np.inf)
        print("OOD_confidence for every sample:::::::", res)
        np.set_printoptions(threshold=1000)
        return res,feat

    def cal_CCA(self, pluggable_dist, test_loader, select_index="CCA"):
        max_sample=self.args.CCA_sample
        feature_extractor=torch.nn.Sequential(*list(self.trainer.model.children())[:-self.args.freeze_layer]).to(self.device)
        feature=[]
        if self.args.if_eval:
            feature_extractor.eval()
        with torch.no_grad():
            for _,(X,_) in enumerate(test_loader):
                X=X.to(self.device)
                if self.args.freeze_layer < len(list(self.trainer.model.children())):
                    feature.append(feature_extractor(X).cpu().detach().numpy())# train_data include X and y
                else:
                    feature.append(X.cpu().detach().numpy())
        feature=np.concatenate(feature, axis=0)
        feat_test=np.squeeze(feature).reshape(-1, np.prod(feature.shape[1:]))

        feat_plug=list(zip(*pluggable_dist))[2]
        
        feat_test={0:np.array(feat_test)}
        feat_plug={id: np.array(plug) for id, plug in enumerate(feat_plug)}
        
        # print(f"explained_variance_ratio_ in test_feature", pca_test.eigenvalues_)
        # print("num of PCA component with effect in test_feature", sum(pca_test.eigenvalues_ > 0.05))
        
        cca_list=[]
        for i in feat_plug:
            min_sample=min(len(feat_plug[i]), len(feat_test[0]))
            
            pca_test = PCA(n_components=min(64, min_sample))
            feat_test_in_use = pca_test.fit_transform(feat_test[0][:min_sample])
            pca_plug = PCA(n_components=min(64, min_sample))
            feat_plug_in_use = pca_plug.fit_transform(feat_plug[i][:min_sample])
            
            if select_index=="CCA":
                # # CCA impleted in sklearn
                # cca=CCA(n_components=self.args.CCA_component)
                # cca.fit(feat_test_in_use, feat_plug_in_use)
                # X_c, Y_c = cca.transform(feat_test_in_use, feat_plug_in_use)
                # print(f"feat_plug_in_use {i} after CCA transform", X_c)
                # corr = np.mean([np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(self.args.CCA_component)])
                
                # SVCCA
                feat_test_in_use_svcca=np.transpose(feat_test_in_use)
                feat_plug_in_use_svcca=np.transpose(feat_plug_in_use)
                corr=svcca(feat_plug_in_use_svcca, feat_test_in_use_svcca, keep_dims=min(min_sample-1, 20))
                
                cca_list.append(corr)
                
            elif select_index=="linear-CKA":
                gram_test = gram_linear(feat_test_in_use)
                gram_plug = gram_linear(feat_plug_in_use)
                cka_from_examples = cka(gram_test, gram_plug)
                cca_list.append(cka_from_examples)
            elif select_index=="rbf-CKA":
                cka_from_examples = cka(gram_rbf(feat_plug_in_use), gram_rbf(feat_test_in_use))
                cca_list.append(cka_from_examples)
            elif select_index=="linear-CKA_debias":
                cka_from_examples = cka(gram_linear(feat_plug_in_use), gram_linear(feat_test_in_use), debiased=True)
                cca_list.append(cka_from_examples)
            elif select_index=="rbf-CKA_debias":
                cka_from_examples = cka(gram_rbf(feat_plug_in_use), gram_rbf(feat_test_in_use), debiased=True)
                cca_list.append(cka_from_examples)
            print(f"number of samples used in CCA for client {i}", min_sample)
        return np.array(cca_list)

    def cal_MMD(self, pluggable_dist, test_loader):
        
        feature_extractor=torch.nn.Sequential(*list(self.trainer.model.children())[:-self.args.freeze_layer]).to(self.device)
        print("feature_extractor.children", list(feature_extractor.children()))
        if self.args.if_eval:
            feature_extractor.eval()
        # print("self.train_data_local_dict", self.train_data_local_dict)
        feature=[]
        # print("self.train_data_local_dict[client_index]====",self.train_data_local_dict[client_index])
        with torch.no_grad():
            for _,(X,_) in enumerate(test_loader):
                X=X.to(self.device)
                print("X:", X.shape)
                if self.args.freeze_layer < len(list(self.trainer.model.children())):
                    feature.append(feature_extractor(X).cpu())# train_data include X and y
                else:
                    feature.append(X.cpu())
        # print("len(feature)",len(feature))
        # print("feature[0].shape===============", feature[0].shape)
        feature=np.concatenate(feature,axis=0)
        # print("feature.shape===============", feature.shape)
        feature=np.squeeze(feature).reshape(-1, np.prod(feature.shape[1:]))
        mean=np.mean(feature,axis=0).flatten() # suppose the sampling dimension is 0
        # print("mean.shape========",mean.shape)
        var=1/feature.shape[0]*((feature-mean).T@(feature-mean))
        # print("var.shape=========",var.shape)
        if self.args.group_feat:
            group = 3
            feature = (feature[:len(feature)//group*group].reshape(-1,group,*feature.shape[1:])).mean(axis=1)
        if self.args.noise_on_test_feat:
            if self.args.noise_type=="stat":
                eye_coef=0.2
                for trial in range(50):
                    try:
                        m=torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(mean), torch.tensor(var*np.eye(var.shape[0]).astype(np.float32)+eye_coef*np.eye(var.shape[0]).astype(np.float32)))
                        break
                    except ValueError:
                        print("generate noise failed, eye_coef:", eye_coef)
                        eye_coef*=2
            elif self.args.noise_type=="random":
                m=torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros_like(torch.tensor(mean)), torch.eye(var.shape[0]))
            feature+=self.args.noisy_coefficient*np.array(m.sample_n(len(feature)))
            feature/=(self.args.noisy_coefficient+1)
        
        # feature=np.concatenate(feature, axis=0)
        # feat_test=np.squeeze(feature).reshape(-1, np.prod(feature.shape[1:]))
        feat_test = feature

        feat_plug=list(zip(*pluggable_dist))[2]
        for id, feat in enumerate(feat_plug):
            print(f"for id:{id}, feat.shape={feat.shape}")
        
        feat_test={0:torch.tensor(feat_test)}
        feat_plug={id: torch.tensor(plug) for id, plug in enumerate(feat_plug)}
        
        
        print("feat_test", feat_test[0])
        print("feat_test.shape after PCA", feat_test[0].shape)
        MMD_list=[]
        for i in feat_plug:
            # min_sample=min(len(feat_plug[i]), len(feat_test[0]))
            # pca_test = PCA(n_components=min(64, min_sample))
            # feat_test_in_use = pca_test.fit_transform(feat_test[0][:min_sample])
            # pca_plug = PCA(n_components=min(64, min_sample))
            # feat_plug_in_use = pca_plug.fit_transform(feat_plug[i][:min_sample])
            
            feat_test_in_use = feat_test[0]
            feat_plug_in_use = feat_plug[i]
            # feat_test_in_use = feat_test[0][:min_sample]
            # feat_plug_in_use = feat_plug[i][:min_sample]
            print("feat_test_in_use", feat_test_in_use.shape)
            print("feat_plug_in_use", feat_plug_in_use.shape)
            
            # print(f"explained_variance_ratio_ in plug{i}", pca_plug.eigenvalues_)
            # print(f"num of PCA component with effect in plug{i}_feature", sum(pca_plug.eigenvalues_ > 0.05))
            # feat_plug_in_use = feat_plug_in_use[np.random.choice(len(feat_plug_in_use), min_sample, replace=False)]
            
            # feat_plug_in_use = np.random.randn(*feat_plug_in_use.shape)*100
            # feat_test_in_use = feat_test[0][np.random.choice(len(feat_test[0]), min_sample, replace=False)]
            
            # Y= np.random.randn(feat_plug_in_use.shape[0], feat_plug_in_use.shape[-1])
            mmd=MMD()
            mmd_value=mmd(feat_test_in_use, feat_plug_in_use).item()
            # print("X_c.shape", X_c.shape)
            # print("np.corrcoef(X_c[:, i], Y_c[:, i])", np.corrcoef(X_c[:, 0], Y_c[:, 0]))
            # print("np.corrcoef(X_c[:, i], Y_c[:, i])", np.corrcoef(X_c[:, 0], Y_c[:, 0]).shape)
            MMD_list.append(mmd_value)
        return MMD_list
                
    def cal_OOD(self, client_list, test_loader):
        logits_list=[]
        # get logits from all plug
        for idx,client in enumerate(client_list):
            if self.args.OOD_independ_classifier:
                model=client.trainer.OOD_dete.to(self.device)
            else:
                model=client.trainer.model.to(self.device)
            if self.args.if_eval:
                model.eval()
            logit=[]
            with torch.no_grad():
                for batch_idx, (x, labels) in enumerate(test_loader):
                    x = x.to(self.device)
                    labels = labels.to(self.device)
                    real_batch_size = labels.shape[0]
                    if self.args.model_input_channels == 3 and x.shape[1] == 1:
                        x = x.repeat(1, 3, 1, 1)
                    if self.args.model_out_feature:
                        output, feat = model(x)
                    else:
                        output = model(x)
                    logit.append(output.cpu().detach().numpy())
            logits=np.concatenate(logit,axis=0)
            # print("logits.shape:::::", logits.shape)
            logits=np.squeeze(logits)
            logits_list.append(torch.Tensor(logits))
        if self.args.OOD_independ_classifier:
            logits_array=torch.softmax(torch.stack(logits_list),dim=-1)
            # print("logits_array.shape:::::", logits_array.shape)
            res=logits_array[...,1].numpy()
            res=np.mean(res, axis=-1)
            # print("res.shape:::::::::", res.shape)
        else:
            logits_array=np.array(logits_list)[...,-1]
            # print("logits_array.shape:::::", logits_array.shape)
            res=torch.sigmoid(torch.Tensor(logits_array)).numpy()
            res=np.mean(res, axis=-1)
            # print("res.shape:::::::::", res.shape)

        return res

    def reduce_plugs(self):
        reduced_idx = set()
        for idx, train_data in enumerate(self.train_data_local_dict):
            if idx not in reduced_idx:
                MMD_list = self.cal_MMD(self.dist_list, train_data)
                normed_MMD_list = (MMD_list-min(MMD_list)) / (max(MMD_list) - min(MMD_list))
                print("normed_MMD_list", normed_MMD_list)
                print("np.where(normed_MMD_list<0.25)", np.where(normed_MMD_list<0.25))
                print("np.setdiff1d(np.where(normed_MMD_list<0.25), [idx])", np.setdiff1d(np.where(normed_MMD_list<0.25), [idx]))
                reduced_idx.update(np.setdiff1d(np.where(normed_MMD_list<0.25), [idx]))
        print("reduced_idx", reduced_idx)
        self.reduce_map = list(range(len(self.dist_list))).pop(reduced_idx)
        
        # self.reduced_dist_list = list.pop(self.dist_list, reduced_idx)
        # self.reduced_model_list = list.pop(self.model_list, reduced_idx)
        
        