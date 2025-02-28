import copy
import logging
import os
import sys
from abc import ABC, abstractmethod

import random

import numpy as np
import torch
import wandb


from utils.perf_timer import Perf_Timer
from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.logger import Logger
from utils.data_utils import (
    get_avg_num_iterations,
    get_label_distribution,
    get_selected_clients_label_distribution
)
from utils.checkpoint import setup_checkpoint_config, save_checkpoint

# from data_preprocessing.build import load_data
from data_preprocessing.build import load_data

from timers.server_timer import ServerTimer
from loss_fn.losses import OOD_loss
from utils.metrics import OOD_Metrics
from optim.build import create_optimizer


track_time = True

from ..fedavg.client import FedAVGClient
import itertools
import gc
# from memory_profiler import profile
from utils.set_seed import set_seed


# @classmethod
# def __file__():
#     pass
class file:
    def __init__(self) -> None:
        pass

def show_memory():
    print("*"*60)
    object_list=[]
    torch.classes.__file__="torch.classes"
    torch.ops.profiler.__file__="torch.ops.profiler"
    torch.ops.quantized.__file__="torch.ops.quantized"
    torch.ops.__loader__.module_repr="torch.ops.quantized"
    for key, value in torch.ops.__dict__.items():
        torch.ops.__getattr__(key).__file__="torch.ops.default"
        torch.ops.__getattr__(key).module_repr="torch.ops.default"
        torch.ops.__getattr__(key).origin="torch.ops.default"
        
    # torch.classes.__file__.__file__="torch_class"
    for obj in gc.get_objects():
        size=sys.getsizeof(obj)
        object_list.append((obj,size))
    sorted_values=sorted(object_list,key=lambda x: x[1],reverse=True)
    with open('memory_profiler.log','w+') as log:
        for obj, size in sorted_values[:10]:
            print(f"ID:{id(obj)}"
                f"TYPE:{type(obj)},"
                f"SIZE:{size/1024/1024:.2f}MB,"
                f"REPR:{str(obj)[:100]}", file=log)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class BasePSManager(object):
    def __init__(self, device, args):
        self.device = device
        self.args = args
        set_seed(self.args.seed)
        # ================================================
        self._setup_datasets()
        set_seed(0)
        self.perf_timer = Perf_Timer(
            verbosity_level=1 if track_time else 0,
            log_fn=Logger.log_timer
        )
        self.selected_clients = None
        self.client_list = []
        self.metrics = Metrics([1], task=self.args.task)
        # ================================================
        if self.args.instantiate_all:
            self.number_instantiated_client = self.args.client_num_in_total
        else:
            self.number_instantiated_client = self.args.client_num_per_round
        self._setup_clients()
        self._setup_server()
        # aggregator will be initianized in _setup_server()
        self.max_comm_round = self.aggregator.get_max_comm_round()
        self.global_num_iterations = self.aggregator.global_num_iterations
        # ================================================
        self.server_timer = ServerTimer(
            self.args,
            self.global_num_iterations,
            local_num_iterations_dict=None
        )
        self.total_train_tracker = RuntimeTracker(
            mode='Train',
            things_to_metric=self.metrics.metric_names,
            timer=self.server_timer,
            args=args
        )
        self.total_test_tracker = RuntimeTracker(
            mode='Test',
            things_to_metric=self.metrics.metric_names,
            timer=self.server_timer,
            args=args
        )
        # ================================================


    def _setup_datasets(self):
        # dataset = load_data(self.args, self.args.dataset)

        dataset = load_data(
                load_as="training", args=self.args, process_id=0, mode="standalone", task="federated", data_efficient_load=True,
                dirichlet_balance=False, dirichlet_min_p=None,
                dataset=self.args.dataset, datadir=self.args.data_dir,
                partition_method=self.args.partition_method, partition_alpha=self.args.partition_alpha,
                client_number=self.args.client_num_in_total, batch_size=self.args.batch_size, num_workers=self.args.data_load_num_workers,
                data_sampler=self.args.data_sampler,
                resize=self.args.dataset_load_image_size, augmentation=self.args.dataset_aug)

        [train_data_num, test_data_num, train_data_global, test_data_global,
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params] = dataset
        self.other_params = other_params
        self.train_global = train_data_global
        self.test_global = test_data_global

        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.class_num = class_num

        if self.args.task in ["classification"] and \
            self.args.loss_fn in ["LDAMLoss", "FocalLoss", "local_FocalLoss", "local_LDAMLoss"]:
            self.local_cls_num_list_dict, self.total_cls_num = get_label_distribution(self.train_data_local_dict, class_num)
        else:
            self.local_cls_num_list_dict = None
            self.total_cls_num = None
        if "traindata_cls_counts" in self.other_params:
            self.traindata_cls_counts = self.other_params["traindata_cls_counts"]
            # Adding missing classes to list
            classes = list(range(self.class_num))
            for key in self.traindata_cls_counts:
                if len(classes) != len(self.traindata_cls_counts[key]):
                    # print(len(classes))
                    # print(len(train_data_cls_counts[key]))
                    add_classes = set(classes) - set(self.traindata_cls_counts[key])
                    # print(add_classes)
                    for e in add_classes:
                        self.traindata_cls_counts[key][e] = 0
        else:
            self.traindata_cls_counts = None

    def _setup_server(self):
        pass

    def _setup_clients(self):
        pass

    def check_end_epoch(self):
        return (self.server_timer.global_outer_iter_idx > 0 and \
            self.server_timer.global_outer_iter_idx % self.global_num_iterations == 0)

    def check_test_frequency(self):
        return self.server_timer.global_outer_epoch_idx % self.args.frequency_of_the_test == 0 or \
            self.server_timer.global_outer_epoch_idx == self.args.max_epochs - 1


    def check_and_test(self):
        if self.check_end_epoch():
            if self.check_test_frequency():
                if self.args.exchange_model:
                    self.test()
                else:
                    self.test_all_clients_model(
                        self.epoch, self.aggregator.model_dict,
                        tracker=self.total_test_tracker, metrics=self.metrics)
            else:
                self.total_train_tracker.reset()
                self.total_test_tracker.reset()
    
    def test(self):
        logging.info("################test_on_server_for_all_clients : {}".format(
            self.server_timer.global_outer_epoch_idx))
        
        self.aggregator.test_on_server_for_all_clients(
            self.server_timer.global_outer_epoch_idx, self.total_test_tracker, self.metrics, self.server_timer.global_comm_round_idx)

        self.total_train_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)
        self.total_test_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)

    def test_all_clients_model(self, epoch, model_dict, tracker=None, metrics=None):
        logging.info("################test_on_server_for_all_clients : {}".format(epoch))
        for idx in model_dict.keys():
            self.aggregator.set_global_model_params(model_dict[idx])
            self.aggregator.test_on_server_for_all_clients(
                epoch, self.total_test_tracker, self.metrics)
        self.total_train_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)
        self.total_test_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)
    
    def test_selection(self, index):
        if index=="CCA":
            index_value=self.aggregator.cal_CCA(self.aggregator.dist_list, self.test_global)
        elif index=="OOD":
            index_value=self.aggregator.cal_OOD(self.client_list, self.test_global)
        print(index_value)
        index_list=np.argsort(index_value)
        print(index_list)
        # print(type(index_list))
        index_value=index_value[index_list]

        ACC_list=[]
        X, y=list(zip(*list(self.test_global)))
        X=torch.cat(X)
        y=torch.cat(y)
        for idx, client in enumerate(self.client_list):
            if index=="CCA":
                # acc=client.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
                test_data=[(torch.tensor(X[0:100]), torch.LongTensor(y[0:100]))]    
                acc=client.trainer.test(test_data, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
            elif index=="OOD":
                acc=client.trainer.OOD_test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
            ACC_list.append(acc)
        ACC_list=np.array(ACC_list)
        print("index_list", index_list)
        print(f"{index}_ON_PLUGGABLE", index_value)
        print(f"ACC_ON_PLUGGABLE_INDEXED_WITH_{index}",ACC_list[index_list])
        wandb.run.summary[f"PLUG_ACC_INDEXED_WITH_{index}"]=ACC_list[index_list]

    def test_selection_on_train(self, index):
        if index=="CCA":
            index_value=self.aggregator.cal_CCA(self.aggregator.dist_list, self.train_global)
        elif index=="OOD":
            index_value=self.aggregator.cal_OOD(self.client_list, self.train_global)
        print(index_value)
        index_list=np.argsort(index_value)
        print(index_list)
        # print(type(index_list))
        index_value=index_value[index_list]
        X, y=list(zip(*list(self.train_global)))
        X=torch.cat(X)
        y=torch.cat(y)
        ACC_list=[]
        for idx, client in enumerate(self.client_list):
            if index=="CCA":
                # acc=client.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
                # print("self.train_global.dataset.data.dtype", self.train_global.dataset.data.dtype)
                # print("X.dtype", X.dtype)
                test_data=[(torch.Tensor(X[0:100]), torch.LongTensor(y[0:100]))]
                # print(test_data[0][0].shape)
                # print(test_data[0][1].shape)
                acc=client.trainer.test(test_data, device=self.device, tracker=self.total_train_tracker, metrics=self.metrics)
            elif index=="OOD":
                acc=client.trainer.OOD_test(self.train_global, device=self.device, tracker=self.total_train_tracker, metrics=self.metrics)
            ACC_list.append(acc)
        ACC_list=np.array(ACC_list)
        print("index_list", index_list)
        print(f"{index}_ON_PLUGGABLE", index_value)
        print(f"ACC_ON_PLUGGABLE_INDEXED_WITH_{index}_ON_TRAIN",ACC_list[index_list])
        wandb.run.summary[f"PLUG_ACC_INDEXED_WITH_{index}_ON_TRAIN"]=ACC_list[index_list]

    def test_locally(self):
        ACC_list=[]
        for idx,client in enumerate(self.client_list):
            acc=client.trainer.test(self.test_data_local_dict[idx], device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
            ACC_list.append(acc)
        print("ACC_ON_LOCAL_PLUGGABLE",ACC_list)
        wandb.run.summary["LOCAL_PLUGGABLE"]=ACC_list

    def test_ACC_of_OOD(self):
        with torch.no_grad():
            if self.args.OOD_noise_option=="other_real_data":
                for client_index,client in enumerate(self.client_list):
                    OOD_dete=client.trainer.OOD_dete.to(device=self.device)
                    real_out=[]
                    real_label=[]
                    x=self.train_data_local_dict[client_index]
                    for idx,(X,y) in enumerate(x):
                        X=X.to(self.device)
                        real_out.append(OOD_dete(X))
                        real_label.append(y)
                    real_out=torch.cat(real_out, dim=0).to(self.device)
                    real_label=torch.cat(real_label, dim=0).to(self.device)
                    noise_out=[]
                    for i in self.train_data_local_dict.keys():
                        if i != client_index:
                            for idx,(X,y) in enumerate(self.train_data_local_dict[i]):
                                X=X.to(self.device)
                                noise_out.append(OOD_dete(X))
                    noise_out=torch.cat(noise_out, dim=0).to(self.device)
                    noise_label=torch.zeros((len(noise_out),)).to(dtype=torch.long).to(self.device)
                    OOD=torch.cat([real_out, noise_out])
                    OOD_label=torch.cat([real_label[...,1], noise_label])
                    res=torch.argmax(OOD,dim=-1)
                    torch.set_printoptions(profile="full")
                
                    print("OOD_labels============", OOD_label)
                    print("res_labels==========", res)
                    torch.set_printoptions(profile="default")
                    print(f"OOD_acc_on_train_for_client{client_index}:::::::", torch.sum(res==OOD_label)/len(OOD_label))
                    
    # load pluggable
    def load_pluggable_for(target):
        def load_pluggable(func):
            def warp(self,*args, **kwargs):
                print(f"load_pluggable_with_{args[0]}==============================")
                # signal_list=[]
                if target=="client":
                    for idx,client in enumerate(self.client_list[0:1]):
                        # signal_list.append(self.aggregator.select_pluggable(client))
                        # self.aggregator.model_list[self.aggregator.select_pluggable(idx)]
                        # print("test_result before download:::::::::::", client.trainer.test(self.test_data_local_dict[idx])))
                        # print(client.trainer.model)
                        selection = self.aggregator.select_pluggable(self.test_data_local_dict[idx], args[0])
                        plug_id = selection[-1] if isinstance(selection, tuple) else selection
                        plug=self.aggregator.model_list[plug_id]
                        for id, layer in enumerate(list(client.trainer.model.children())[-self.args.freeze_layer:]):
                            layer.load_state_dict(plug[id].state_dict())
                        # client.trainer.model=torch.nn.Sequential(*(list(client.trainer.model.children())[:-1]),*self.aggregator.model_list[self.aggregator.select_pluggable(idx)])
                        # print(client.trainer.model)
                        # print("test_result after download:::::::::::", client.trainer.test(self.test_data_local_dict[idx]))
                    print("load pluggable successfully==============================")
                elif target=="batch":
                    for idx, client in enumerate(self.client_list):
                        if not self.args.non_iid_for_batch_data:
                            test_data=self.niid_loader[len(self.niid_loader)//len(self.client_list)*idx:len(self.niid_loader)//len(self.client_list)*(idx+1)]
                        else:
                            # print([i for i in self.net_dataidx_map[idx]])
                            test_data=[self.niid_loader[i] for i in self.net_dataidx_map[idx]]
                            # print("len(test_data)", len(test_data))
                            # print("test_data[0][0].shape",test_data[0][0].shape)
                            # print("test_data[0][1].shape",test_data[0][1].shape)
                        selection = self.aggregator.select_pluggable(self.test_data_local_dict[idx], args[0])
                        plug_id = selection[-1] if isinstance(selection, tuple) else selection
                        plug=self.aggregator.model_list[plug_id]
                        for id, layer in enumerate(list(client.trainer.model.children())[-self.args.freeze_layer:]):
                            layer.load_state_dict(plug[id].state_dict())
                elif target=="Personalized batch":
                    choice={}
                    score_list=[]
                    for idx, client in enumerate(self.client_list):
                        # for all client
                        # test_data=[new_loader[i] for i in net_dataidx_map[idx]]
                        # plug_list=self.aggregator.select_pluggable(test_data, args[0])
                        # plug=self.aggregator.model_list[plug_list[idx]]
                        # one client by another
                        # print(f"PICKING PLUGGABLE FOR CLIENT{idx}")
                        # print("len(self.p_loader)====", len(self.p_loader))
                        test_data=[self.p_loader[i] for i in self.personalized_map[idx]]
                        if args[0]=="index":
                            plug=self.aggregator.select_pluggable(test_data, args[0])
                            plug=plug[idx]
                        elif args[0]=="personalized_with_OOD":
                            score, plug=self.aggregator.select_pluggable(test_data, args[0], self.client_list)
                            score_list.append(score)
                        else:
                            score, plug=self.aggregator.select_pluggable(test_data, args[0])
                            score_list.append(score)
                        print(f"client{idx} is choosing plug{plug}")
                        choice[str(idx)]=plug
                        plug=self.aggregator.model_list[plug]
                        for id, layer in enumerate(list(client.trainer.model.children())[-self.args.freeze_layer:]):
                            layer.load_state_dict(plug[id].state_dict())
                    score = np.array(score_list)
                    name='choice_'+str(args[0])
                    score_name='score_'+str(args[0])
                    np.save(f"result/{self.args.client_num_in_total}client_{self.args.noisy_coefficient}noise_{self.args.noise_type}type_{self.args.freeze_layer}layer_{self.args.partition_alpha}alpha_{self.args.dataset}_{score_name}.npy", score)
                    if self.args.reduce_plug:
                        np.save(f"result/{self.args.client_num_in_total}client_{self.args.noisy_coefficient}noise_{self.args.noise_type}type_{self.args.freeze_layer}layer_{self.args.partition_alpha}alpha_{self.args.dataset}_reduce_map.npy", self.aggregator.reduce_map)
                    wandb.run.summary[name]=choice
                    wandb.run.summary[score_name]=score
                return func(self,*args, **kwargs)
            return warp
        return load_pluggable

    def generate_non_iid_batch(self):
        if self.args.non_iid_for_batch_data:
            X, y=list(zip(*list(self.test_global)))
            labels=torch.cat(y).numpy()
            X=torch.cat(X).unsqueeze(1)
            y=torch.cat(y).unsqueeze(1)
            new_loader=list(zip(X,y))
            self.niid_loader=new_loader
            min_size = 0
            while min_size < self.args.CCA_sample:
                idx_batch = [[] for _ in range(len(self.client_list))]
                for k in range(self.args.num_classes):
                    idx_k = np.where(labels == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.args.partition_alpha, len(self.client_list)))
                    proportions = np.array([p * (len(idx_j) < len(labels) / len(self.client_list)) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    # print("proportions", proportions)
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    # print("len(idx_batch)", len(idx_batch))
                print("[len(idx_j) for idx_j in idx_batch ]", [len(idx_j) for idx_j in idx_batch])
                print("sum of len(idx_j)", sum([len(idx_j) for idx_j in idx_batch]))
                min_size = min([len(idx_j) for idx_j in idx_batch])
            net_dataidx_map={}
            for j in range(len(self.client_list)):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]
            self.net_dataidx_map=net_dataidx_map

    def generate_personalized_data(self):
        X, y=list(zip(*list(self.test_global)))
        X=torch.cat(X).unsqueeze(1)
        y=torch.cat(y).unsqueeze(1)
        if self.args.if_OOD and self.args.OOD_independ_classifier:
            train_label=[i.dataset.tensors[1][...,0].numpy() for i in self.train_data_local_dict.values()]
            labels=np.array(y).reshape(-1,2)[...,0] # classify label and OOD label
        else:
            # X, y=list(zip(*list(self.train_data_local_dict[i])))
            train_label=[i.dataset.targets for i in self.train_data_local_dict.values()]
            labels=np.array(y).reshape(-1,1)
        print(train_label[0].shape)
        class_propotion=np.array([[np.sum(y==i) for i in range(self.args.num_classes)] for y in train_label])
        # print(class_propotion)
        num_train=np.sum(class_propotion)
        num_class=np.sum(class_propotion, axis=0, keepdims=False)
        
        # print("num_class====", num_class)
        # class_propotion=class_propotion/num_test
        new_loader=list(zip(X,y))
        self.p_loader=new_loader
        num_test=len(labels)
        min_size=0

        idx_batch = [[] for _ in range(len(self.client_list))]
        for k in range(self.args.num_classes):
            idx_k = np.where(labels == k)[0]
            # print("len(idx_k)", len(idx_k))
            num=len(idx_k)
            np.random.shuffle(idx_k)
            k_num=(np.cumsum(class_propotion[:, k]*1.0/num_class[k])*num).astype(int)[:-1]
            # print("k_num:", k_num)
            # print("spilt result::::",np.split(idx_k, k_num))
            idx_batch=[idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, k_num))]
        #     print("len(idx_batch)", len(idx_batch))
        # print("[len(idx_j) for idx_j in idx_batch]", [len(idx_j) for idx_j in idx_batch])
        # print("sum of len(idx_j)", sum([len(idx_j) for idx_j in idx_batch]))
        min_size = min([len(idx_j) for idx_j in idx_batch])
        
        net_dataidx_map={}

        for j in range(len(self.client_list)):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        self.personalized_map=net_dataidx_map
        
        testdata_cls_number={}
        for idx in range(len(self.client_list)):
            test_data = [self.p_loader[i] for i in self.personalized_map[idx]]
            X, y = list(zip(*test_data))
            if not self.args.if_OOD:
                y=np.array(y)
            else:
                y=np.array([item[0,1] for item in y])
            # print("y", y)
            # print("y==", y==1)
            testdata_cls_number[idx] = {i:sum(y==i) for i in range(self.args.num_classes)}
        testdata_cls_matrix=np.zeros((len(self.client_list), self.args.num_classes))
        for id in testdata_cls_number:
            d_number = testdata_cls_number[id]
            for i in d_number:
                testdata_cls_matrix[id, i] = d_number[i]
                
        np.save(f"result/{self.args.dataset}_{self.args.partition_alpha}alpha_{self.args.client_num_in_total}client_testdata_cls_matrix", testdata_cls_matrix)
        # print("self.personalized_map===", self.personalized_map)

    @load_pluggable_for("client")
    def test_with_selected_pluggable_for_clients(self, select_index):
        logging.info("################test_with_selected_pluggable_for_clients:")
        acc=0
        for idx,client in enumerate(self.client_list[0:1]):
            acc+=client.trainer.test(self.test_data_local_dict[idx], device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
        acc/=len(self.client_list[0:1])*1.0
        print(f"ACC_SELECT_PLUGGABLE_WITH_{select_index}_FOR_CLIENT:::::::::::::::::::::::::::::::::::::::",acc)
        wandb.run.summary[f"select_with_{select_index}_for_client_acc"] = acc
        # self.total_train_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)
        # self.total_test_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)

    @load_pluggable_for("batch")
    def test_with_selected_pluggable_for_batch(self, select_index):
        logging.info("################test_with_selected_pluggable_for_clients:")
        acc=0
        num_sample_a=0
        for idx,client in enumerate(self.client_list):
            if self.args.non_iid_for_batch_data:
                test_data=[self.niid_loader[i] for i in self.net_dataidx_map[idx]]
            else:
                test_data=list(self.test_global)[len(self.test_global)//len(self.client_list)*idx:len(self.test_global)//len(self.client_list)*(idx+1)]
            num_sample_c=len(test_data)
            num_sample_a+=num_sample_c
            X, y=list(zip(*test_data))
            dataset=torch.utils.data.TensorDataset(torch.cat(X),torch.cat(y))
            test_data=torch.utils.data.dataloader.DataLoader(dataset, batch_size=128)
            acc+=client.trainer.test(test_data, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)*num_sample_c
        acc/=num_sample_a*1.0
        print(f"ACC_SELECT_PLUGGABLE_WITH_{select_index}_FOR_BATCH:::::::::::::::::::::::::::::::::::::::",acc)
        wandb.run.summary[f"select_with_{select_index}_for_batch_acc"] = acc

    @load_pluggable_for("Personalized batch")
    def test_with_personalized_batch(self, select_index="index"):
        logging.info("################test_with_personalized_batch:")
        acc=0
        num_sample_a=0
        acc_dict={}
        for idx,client in enumerate(self.client_list):
            test_data=[self.p_loader[i] for i in self.personalized_map[idx]]
            num_sample_c=len(test_data)
            X, y=list(zip(*test_data))
            dataset=torch.utils.data.TensorDataset(torch.cat(X),torch.cat(y))
            test_data=torch.utils.data.dataloader.DataLoader(dataset, batch_size=128)
            num_sample_a+=num_sample_c
            if select_index=="personalized_with_OOD":
                acc_client = client.trainer.OOD_test(test_data, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
                acc += acc_client * num_sample_c
            else:
                acc_client = client.trainer.test(test_data, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
                acc += acc_client * num_sample_c
            acc_dict[idx]=acc_client
        print(f"ACC_DICT FOR PERSONALIZED_BATCH_WITH{select_index}:::::::::::::::::::::::::::::::::::::::", acc_dict)
        acc/=num_sample_a*1.0
        print(f"ACC_FOR_PERSONALIZED_BATCH_WITH{select_index}:::::::::::::::::::::::::::::::::::::::", acc)
        wandb.run.summary[f"personalized_acc_with_{select_index}"] = acc
    import sys, os
        
    def test_with_personalized_batch_with_plugn(self, n):
        logging.info("################test_with_personalized_batch:")
        acc=0
        num_sample_a=0
        acc_dict={}
        plug=self.aggregator.model_list[n]
        for idx,client in enumerate(self.client_list):
            for id, layer in enumerate(list(client.trainer.model.children())[-self.args.freeze_layer:]):
                layer.load_state_dict(plug[id].state_dict())
                    
            # print(f"client{idx}.trainer.model.linear.weight in personalize", client.trainer.model.linear.weight)
            test_data=[self.p_loader[i] for i in self.personalized_map[idx]]
            num_sample_c=len(test_data)
            X, y=list(zip(*test_data))
            print("len(X)", len(X))
            print("len(y)", len(y))
            dataset=torch.utils.data.TensorDataset(torch.cat(X),torch.cat(y))
            test_data=torch.utils.data.dataloader.DataLoader(dataset, batch_size=128)
            num_sample_a+=num_sample_c
            
            print('This will print')
            blockPrint()
            acc_client = client.trainer.test(test_data, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
            print("This won't")
            enablePrint()
            print("This will too")
            
            print(f"num_sample_c in client{idx}", num_sample_c)
            acc += acc_client * num_sample_c
            acc_dict[idx]=acc_client
        print(f"ACC_DICT FOR PERSONALIZED_BATCH_WITH_PLUG{n}:::::::::::::::::::::::::::::::::::::::", acc_dict)
        print(f"num_sample_a", num_sample_a)
        acc/=num_sample_a*1.0
        print(f"ACC_FOR_PERSONALIZED_BATCH_WITH_plug{n}:::::::::::::::::::::::::::::::::::::::", acc)
        wandb.run.summary[f"personalized_acc_with_plug{n}"] = acc
        
    def test_with_global_with_plugs(self):
        logging.info("################test_with_global_dataset:")
        acc_dict={}
        for idx, client in enumerate(self.client_list):
            plug=self.aggregator.model_list[9]
            for id, layer in enumerate(list(client.trainer.model.children())[-self.args.freeze_layer:]):
                layer.load_state_dict(plug[id].state_dict())
            for layer in list(client.trainer.model.children())[:-self.args.freeze_layer]:
                if "Norm" in type(layer).__name__:
                    print("BN parameter mean:::::::::::::::::", layer.running_mean)
                    print("BN parameter var:::::::::::::::::", layer.running_var)
                    print("BN parameter mean.shape:::::::::::::::::", layer.running_mean.shape)
                    print("BN parameter var.shape:::::::::::::::::", layer.running_var.shape)
            # print(f"client{idx}.trainer.model.linear.weight in ", client.trainer.model.linear.weight)
            # print(f"client{idx}.trainer.model.linear.bias in ", client.trainer.model.linear.bias)
            print('This will print')
            blockPrint()
            acc=client.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
            print("This won't")
            enablePrint()
            print("This will too")
            acc_dict[idx]=acc
            print(f"ALL_GLOBAL_DATA ACC FOR CLIENT{idx}", acc)
        print(f"ACC_DICT FOR GLOBAL_DATA:::::::::::::::::::::::::::::::::::::::", acc_dict)
        
        
    def test_with_CCA_between_personalized_batch(self, select_index="index"):
        score_list=[]
        choice={}
        for idx, client in enumerate(self.client_list):
            # one client by another
            print(f"CHECKING CCA FOR CLIENT{idx}")
            test_data=self.aggregator.dist_list[idx]
            score, plug=self.aggregator.test_CCA(test_data, select_index)
            score_list.append(score)
            print(f"client{idx} is choosing plug{plug}")
            choice[str(idx)]=plug
        score = np.array(score_list)
        name='choice_'+select_index
        score_name='score_'+select_index
        np.save(f"result/{score_name}.npy", score)
        wandb.run.summary[name]=choice
        wandb.run.summary[score_name]=score
                
            
    def test_all_samples(self, select_index):
        logging.info("################test_all_samples: ")
        logits_list=[]
        # get logits from all 
        if select_index=="distance":
            index_list, feat=self.aggregator.select_pluggable(self.test_global, "distance")
        elif select_index=="OOD":
            index_list, feat=self.aggregator.select_pluggable(self.test_global, "OOD", self.client_list)
        feat=torch.Tensor(feat).to(self.device)
        print(feat.shape)
        loader=self.test_global
        # print("len(index_list) for everysample", len(index_list))
        # print("self.aggregator.model_list========", self.aggregator.model_list[0])
        # unpack the original batched data
        X,Y=list(zip(*list(loader)))
        X=list(torch.cat(X))
        target=list(torch.cat(Y))
        new_loader=zip(X,target)
        labels=torch.cat(Y).numpy()

        logits=[]
        with torch.no_grad():
            for index, (_, _) in enumerate(new_loader):
                if self.args.freeze_layer==1:
                    model=torch.nn.Sequential(*self.aggregator.model_list[index_list[index]]).to(self.device)
                    if self.args.if_eval:
                        model.eval()
                    output=model(feat[index])
                elif self.args.freeze_layer>1:
                    layer=self.aggregator.model_list[index_list[index]][-self.args.freeze_layer].to(self.device)
                    if self.args.if_eval:
                        layer.eval()
                    out=layer(feat[index].unsqueeze(dim=0))
                    index = self.args.freeze_layer - 1
                    while index>1:
                        layer=self.aggregator.model_list[index_list[index]][-index].to(self.device)
                        if self.args.if_eval:
                            layer.eval()
                        out=layer(out)
                        index-=1
                    layer=self.aggregator.model_list[index_list[index]][-1].to(self.device)
                    if self.args.if_eval:
                        layer.eval()
                    output=layer(out.view(out.size(0), -1))
                    
                logits.append(output.cpu().detach().numpy())
                # print(logits[-1].shape)
        logits=np.stack(logits,axis=0)
        logits=np.squeeze(logits)
        logits_list.append(logits)

        logits_array=np.array(logits_list[0])
        # print(logits_array.shape)
        # print(labels)
        if select_index != "OOD":
            res=torch.nn.functional.softmax(torch.Tensor(logits_array), dim=1).numpy()
        else:
            labels=labels[...,0]
            res=logits_array
            # print("labels of OOD", labels)
        res=np.argmax(res, axis=1)
        acc=np.sum(res==labels)/len(labels)
        print("ACC_ON_SELECT_MODEL_FOR_EVERY_SAMPLE:::::::::::::::::::::::::::::::::::::::",acc*100)
        wandb.run.summary[f"SELECT_MODEL_FOR_EVERY_SAMPLE_acc"] = acc*100

    def test_all_pluggable(self):
        logging.info("################test_all_pluggable: ")
        logits_list=[]
        # get logits from all plug
        for idx,client in enumerate(self.client_list):
            model=client.trainer.model.to(self.device) # ?
            if self.args.if_eval:
                model.eval()
            logit=[]
            with torch.no_grad():
                for batch_idx, (x, labels) in enumerate(self.test_data_local_dict[idx]):
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
            logits=np.squeeze(logits)
            logits_list.append(logits)
        # get all label
        labels_numpy=torch.cat((list(zip(*list(self.test_global)))[1])).numpy()
        logits_array=np.array(logits_list)
        # print("logits_array.shape:::::", logits_array.shape)
        
        res=torch.nn.functional.softmax(torch.Tensor(logits_array), dim=-1).numpy()
        res= np.argmax(res, axis=-1)
        # print("res.shape:::::", res.shape)
        # print((res[:, 0]==labels[0]).any())
        acc= np.sum([(res[:, col]==labels_numpy[col]).any() for col in range(res.shape[-1])])/len(labels_numpy)
        print("ACC_ON_ALL_PLUGGABLE:::::::::::::::::::::::::::::::::::::::", acc*100)
        wandb.run.summary[f"ONE_Correct_acc"] = acc*100
        pass
                
    def test_ensemble(self, ensemble_method):
        logging.info("################test_ensemble: ")
        logits_list=[]
        # get logits from all 
        if ensemble_method=="Avg_all":
            plug_list = self.aggregator.select_pluggable(self.test_global, "index")
        elif ensemble_method=="multi-Pluggable_with_CCA":
            plug_list = self.aggregator.select_pluggable(self.test_global, "multi-Pluggable_with_CCA")
        elif ensemble_method=="multi-Pluggable_with_OOD":
            plug_list = self.aggregator.select_pluggable(self.test_global, "multi-Pluggable_with_OOD", self.client_list)
        for idx in plug_list:
            client=self.client_list[idx]
            model=client.trainer.model.to(self.device)
            if self.args.if_eval:
                model.eval()
            logit=[]
            with torch.no_grad():
                for batch_idx, (x, labels) in enumerate(self.test_global):
                    x = x.to(self.device)
                    labels = labels.to(self.device)
                    if self.args.model_input_channels == 3 and x.shape[1] == 1:
                        x = x.repeat(1, 3, 1, 1)
                    if self.args.model_out_feature:
                        output, feat = model(x)
                    else:
                        output = model(x)
                    logit.append(output.cpu().detach().numpy())
            logits=np.concatenate(logit,axis=0)
            logits=np.squeeze(logits)
            logits_list.append(logits)
        # get all label
        labels=torch.cat((list(zip(*list(self.test_global)))[1])).numpy()
        # print(labels)
        logits_array=np.array(logits_list)
        # logits_array=np.mean(logits_array, axis=0)
        # print(logits_array.shape)
        # print(labels.shape)
        if self.args.if_OOD:
            if self.args.OOD_independ_classifier:
                logits_array=logits_array
            else:
                logits_array=logits_array[:,labels[::,-1]==1, :-1]
            labels=labels[labels[...,-1]==1, 0]
        print("logits_array.shape for OOD::::", logits_array.shape)
        print("labels.shape for OOD:::::", labels.shape)
        # loss = loss_fn(torch.FloatTensor(logits_array), torch.LongTensor(labels))
        res=torch.nn.functional.softmax(torch.Tensor(logits_array), dim=-1).numpy()

        res=np.mean(res, axis=0)
        print("res.shape::::::", res.shape)
        print("label.shape::::::", labels.shape)
        res=np.argmax(res, axis=1)
        print(res)
        print(labels)
        acc=np.sum(res==labels)/len(labels)
        print("ACC_ON_ENSEMBLE_MODEL:::::::::::::::::::::::::::::::::::::::",acc*100)
        wandb.run.summary[f"{ensemble_method}_acc"] = acc*100
    
    def feat_norm(self):
        feature_extractor=torch.nn.Sequential(*list(self.aggregator.trainer.model.children())[:-self.args.freeze_layer]).to(self.device)
        feature=[]
        test_X=self.test_global.dataset.tensors[0]
        test_y=self.test_global.dataset.tensors[1]
        print("test_y.shape===",test_y.shape)
        noise=test_y[...,-1]==0
        real_data=test_y[...,-1]==1
        ori_noise=test_X[noise].reshape(-1, np.prod(test_X.shape[1:]))
        ori_real=test_X[real_data].reshape(-1, np.prod(test_X.shape[1:]))
        norm_noise=np.linalg.norm(ori_noise,ord=2)
        norm_real=np.linalg.norm(ori_real,ord=2)
        print("ori_noise_matrix_norm===", norm_noise)
        print("ori_real_matrix_norm===", norm_real)
        norm_noise_v=np.linalg.norm(ori_noise,ord=2,axis=1)
        norm_real_v=np.linalg.norm(ori_real,ord=2,axis=1)
        print("ori_noise_vec_norm===", norm_noise_v)
        print("ori_real_vec_norm===", norm_real_v)
        if self.args.if_eval:
            feature_extractor.eval()
        with torch.no_grad():
            for _,(X,_) in enumerate(self.test_global):
                X=X.to(self.device)
                if self.args.freeze_layer < len(list(self.aggregator.trainer.model.children())):
                    feature.append(feature_extractor(X).cpu().detach().numpy())# train_data include X and y
                else:
                    feature.append(X.cpu().detach().numpy())
        feature=np.concatenate(feature, axis=0)
        feat=np.squeeze(feature)
        feat_test=np.squeeze(feature).reshape(-1, np.prod(feature.shape[1:]))
        feat_noise=np.linalg.norm(feat_test[noise],ord=2)
        feat_real=np.linalg.norm(feat_test[real_data],ord=2)
        print("feat_noise_matrix_norm===", feat_noise)
        print("feat_real_matrix_norm===", feat_real)
        feat_noise=np.linalg.norm(feat_test[noise],ord=2,axis=1)
        feat_real=np.linalg.norm(feat_test[real_data],ord=2,axis=1)
        print("feat_noise_vec_norm===", feat_noise)
        print("feat_real_vec_norm===", feat_real)
        pass

    def train_centrally(self):
        try:
            params=torch.load(f"{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.args.central_trainning_epoch}_epoch_alpha={self.args.partition_alpha}.pth")
            self.aggregator.trainer.set_model_params(copy.deepcopy(params))
            for index,client in enumerate(self.client_list):
                client.trainer.set_model_params(copy.deepcopy(self.aggregator.trainer.get_model_params()))
            wandb.run.summary[f"pretrain"]=False
        except IOError:   
            dataset = load_data(
                    load_as="training", args=self.args, process_id=0, mode="centralized", task="centralized", data_efficient_load=True,
                    dirichlet_balance=False, dirichlet_min_p=None,
                    dataset=self.args.dataset, datadir=self.args.data_dir,
                    partition_method=self.args.partition_method, partition_alpha=self.args.partition_alpha,
                    client_number=self.args.client_num_in_total, batch_size=self.args.batch_size, num_workers=self.args.data_load_num_workers,
                    data_sampler=self.args.data_sampler,
                    resize=self.args.dataset_load_image_size, augmentation=self.args.dataset_aug)
            train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = dataset
            for i in range(self.args.central_trainning_epoch):
                self.aggregator.trainer.train_one_epoch(train_data=train_dl, device=self.device)
                print("epoch{} , ", i)
            torch.save(self.aggregator.trainer.get_model_params(), f=f"{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.central_trainning_epoch}_epoch_alpha={self.args.partition_alpha}.pth")
            for index,client in enumerate(self.client_list):
                client.trainer.set_model_params(copy.deepcopy(self.aggregator.trainer.get_model_params()))
        acc=self.aggregator.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
        print("ACC_ON_CENTRALLY_TRAINED::::::::::::::::::::::::", acc)
        wandb.run.summary[f"backbone_acc"] = acc
        wandb.run.summary[f"pretrain"]=True

    

    def freeze(self):
        def flatten_submodules(module):
            if type(module)==list:
                submodules = module
            else:
                submodules = list(module.children())
            if not submodules:
                return [module]
            flattened_submodules = []
            for submodule in submodules:
                flattened_submodules += flatten_submodules(submodule)
            return flattened_submodules
        
        # TODO: freeze backbone of self.aggregator.trainer.classifier and all client.trainer.classifier
        print("Model Architechture", self.aggregator.trainer.model)
        for layer in list(self.aggregator.trainer.model.children())[:-self.args.freeze_layer]:
            for i in list(layer.parameters()):
                i.requires_grad=False
        # freeze BN
        for layer in flatten_submodules(list(self.aggregator.trainer.model.children())[:-self.args.freeze_layer]):
            if isinstance(layer, torch.nn.BatchNorm2d) and self.args.freeze_bn:
                logging.info(f"detech Norm Layer {type(layer).__name__}, Freezing..........")
                layer.eval()
        
        for client in self.client_list:
            num_bn=0
            for layer in list(client.trainer.model.children())[:-self.args.freeze_layer]:
                for i in list(layer.parameters()):
                    i.requires_grad=False
            # freeze BN
            flatten_model = flatten_submodules(list(client.trainer.model.children())[:-self.args.freeze_layer])
            for layer in flatten_model:
                if isinstance(layer, torch.nn.BatchNorm2d) and self.args.freeze_bn:
                    num_bn += 1
                    logging.info(f"detech Norm Layer {type(layer).__name__}, Freezing..........")
                    layer.eval()
            print("num_batchnorm:::::", num_bn)
        
        # TODO: seems OK to not freeze backbone of OOD_dete
        # if self.args.if_OOD and self.args.OOD_independ_classifier and self.args.freeze_OOD_backbone:
        #     for layer in list(self.aggregator.trainer.OOD_dete.children())[:-self.args.freeze_layer]:
        #         for i in list(layer.parameters()):
        #             i.requires_grad=False
        #     # freeze BN
        #     for layer in flatten_submodules(list(self.aggregator.trainer.OOD_dete.children())[:-self.args.freeze_layer]):
        #         if isinstance(layer, torch.nn.BatchNorm2d) and self.args.freeze_bn:
        #             logging.info(f"detech Norm Layer {type(layer).__name__}, Freezing..........")
        #             layer.eval()
            
        #     for client in self.client_list:
        #         num_bn=0
        #         for layer in list(client.trainer.OOD_dete.children())[:-self.args.freeze_layer]:
        #             for i in list(layer.parameters()):
        #                 i.requires_grad=False
        #         # freeze BN
        #         flatten_model = flatten_submodules(list(client.trainer.OOD_dete.children())[:-self.args.freeze_layer])
        #         for layer in flatten_model:
        #             if isinstance(layer, torch.nn.BatchNorm2d) and self.args.freeze_bn:
        #                 num_bn += 1
        #                 logging.info(f"detech Norm Layer {type(layer).__name__}, Freezing..........")
        #                 layer.eval()
        #         print("num_batchnorm:::::", num_bn)
    
    def OOD_mod_data(self, origin, ori_type=None):
        if ori_type == "train_data_local_dict" or ori_type =="test_data_local_dict":
            new={}
            for i in origin.keys():
                X, y=list(zip(*list(origin[i])))
                X=torch.cat(X)
                if not self.args.OOD_feat_noise:
                    OOD_data=torch.rand(X.size())
                    X=torch.cat([X,OOD_data])
                idx = torch.randperm(len(X))
                X = X[idx]
                y=torch.cat(y).unsqueeze(-1)
                y=torch.cat([y, torch.ones((len(y),1))],dim=1)
                if not self.args.OOD_feat_noise:
                    OOD_label=torch.zeros(y.size())
                    y=torch.cat([y, OOD_label])
                y = y[idx]
                print("X.shape=======", X.shape)
                print("y.shape=======", y.shape)
                dataset=torch.utils.data.TensorDataset(X,y)
                new[i]=torch.utils.data.dataloader.DataLoader(dataset, batch_size=self.args.batch_size)
        else:
            X, y=list(zip(*list(origin)))
            X=torch.cat(X)
            if not self.args.OOD_feat_noise:
                OOD_data=torch.rand(X.size())
                X=torch.cat([X,OOD_data])
            # shuffle new data 
            idx = torch.randperm(len(X))
            X = X[idx]
            y=torch.cat(y).unsqueeze(-1)
            y=torch.cat([y, torch.ones((len(y),1))],dim=1)
            if not self.args.OOD_feat_noise:
                OOD_label=torch.zeros(y.size())
                y=torch.cat([y, OOD_label])
            # shuffle new label
            y = y[idx]
            print("X.shape=======", X.shape)
            print("y.shape=======", y.shape)
            dataset=torch.utils.data.TensorDataset(X,y)
            new=torch.utils.data.dataloader.DataLoader(dataset, batch_size=self.args.batch_size)
        return new

    def OOD_remodel(self):
        params_before=copy.deepcopy(list(self.aggregator.trainer.model.children()))
        params_l_before=params_before[-1].state_dict()
        params_shape=params_l_before['weight'].shape
        num_feat=params_shape[-1]
        if self.args.OOD_independ_classifier:
            self.aggregator.trainer.OOD_dete=copy.deepcopy(self.aggregator.trainer.model)
            self.aggregator.trainer.OOD_dete.linear=torch.nn.Linear(num_feat, 2)
            torch.nn.init.xavier_normal_(self.aggregator.trainer.OOD_dete.linear.weight)
            torch.nn.init.constant_(self.aggregator.trainer.OOD_dete.linear.bias, 0)
        else:
            # load layers before
            self.aggregator.trainer.model.linear=torch.nn.Linear(num_feat, self.args.num_classes+1)
            for idx, layer in enumerate(list(self.aggregator.trainer.model.children())[-self.args.freeze_layer:-1]):
                layer.load_state_dict(copy.deepcopy(params_before[idx].state_dict()))
            # load linear
            torch.nn.init.xavier_normal_(list(self.aggregator.trainer.model.children())[-1].weight)
            torch.nn.init.constant_(list(self.aggregator.trainer.model.children())[-1].bias, 0)
            list(self.aggregator.trainer.model.children())[-1].weight=torch.nn.parameter.Parameter(torch.cat([params_l_before['weight'], list(self.aggregator.trainer.model.children())[-1].weight[-1:]]))
            list(self.aggregator.trainer.model.children())[-1].bias=torch.nn.parameter.Parameter(torch.cat([params_l_before['bias'], list(self.aggregator.trainer.model.children())[-1].bias[-1:]]))
        
        self.train_global=self.OOD_mod_data(self.train_global)
        self.test_global=self.OOD_mod_data(self.test_global)
        self.train_data_local_dict=self.OOD_mod_data(self.train_data_local_dict, "train_data_local_dict")
        self.test_data_local_dict=self.OOD_mod_data(self.test_data_local_dict, "test_data_local_dict")

        params=None
        self.metrics=OOD_Metrics([1], task=self.args.task, OOD_independ_classifier=self.args.OOD_independ_classifier)
        self.aggregator.train_data_local_dict=copy.deepcopy(self.train_data_local_dict)
        self.aggregator.test_data_local_dict=copy.deepcopy(self.test_data_local_dict)
        self.aggregator.train_global=copy.deepcopy(self.train_global)
        self.aggregator.test_global=copy.deepcopy(self.test_global)
        self.aggregator.trainer.criterion=OOD_loss(OOD_independ_classifier=self.args.OOD_independ_classifier)
        self.aggregator.trainer.metrics=OOD_Metrics([1], task=self.args.task, OOD_independ_classifier=self.args.OOD_independ_classifier)
        self.aggregator.metrics=OOD_Metrics([1], task=self.args.task, OOD_independ_classifier=self.args.OOD_independ_classifier)
        self.aggregator.trainer.optimizer=create_optimizer(self.args, self.aggregator.trainer.model, params=params, role='server')
        if self.args.OOD_independ_classifier:
            self.aggregator.trainer.c_optimizer=create_optimizer(self.args, self.aggregator.trainer.OOD_dete, params=params, role='server')
        for index,client in enumerate(self.client_list):
            client.trainer.model=copy.deepcopy(self.aggregator.trainer.model)
            client.trainer.criterion=OOD_loss(OOD_independ_classifier=self.args.OOD_independ_classifier)
            client.trainer.metrics=OOD_Metrics([1], task=self.args.task, OOD_independ_classifier=self.args.OOD_independ_classifier)
            client.metrics=OOD_Metrics([1], task=self.args.task, OOD_independ_classifier=self.args.OOD_independ_classifier)
            client.train_data_local_dict=copy.deepcopy(self.train_data_local_dict)
            client.trainer.optimizer=create_optimizer(self.args, client.trainer.model, params=params, role='client')
            if self.args.OOD_independ_classifier:
                client.trainer.OOD_dete=copy.deepcopy(self.aggregator.trainer.OOD_dete)
                client.trainer.c_optimizer=create_optimizer(self.args, client.trainer.OOD_dete, params=params, role='client')

    def get_init_state_kargs(self):
        self.selected_clients = [i for i in range(self.args.client_num_in_total)]
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss", "local_FocalLoss", "local_LDAMLoss"]:
            self.selected_clients_label_distribution = get_selected_clients_label_distribution(
                self.local_cls_num_list_dict, self.class_num, self.selected_clients, min_limit=1)
            init_state_kargs = {"weight": None, "selected_cls_num_list": self.selected_clients_label_distribution,
                                "local_cls_num_list_dict": self.local_cls_num_list_dict}
        else:
            init_state_kargs = {}
        return init_state_kargs

    def get_update_state_kargs(self):
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss", "local_FocalLoss", "local_LDAMLoss"]:
            self.selected_clients_label_distribution = get_selected_clients_label_distribution(
                self.local_cls_num_list_dict, self.class_num, self.selected_clients, min_limit=1)
            update_state_kargs = {"weight": None, "selected_cls_num_list": self.selected_clients_label_distribution,
                                "local_cls_num_list_dict": self.local_cls_num_list_dict}
        else:
            update_state_kargs = {}
        return update_state_kargs

    def lr_schedule(self, num_iterations, warmup_epochs):
        epochs = self.server_timer.global_outer_epoch_idx
        iterations = self.server_timer.global_outer_iter_idx

        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.aggregator.trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.aggregator.trainer.lr_schedule(epochs)

    # @profile(stream=open('memory_profiler.log','w+'))
    # ==============train clients and add results to aggregator ===================================
    def train(self):
        if self.args.comm_round > 0:
            # change initial freeze_bn to train model with normal bn
            self.args.freeze_bn=False
            try:
                
                # FedRod + HPFL , remember to remove after test
                if self.args.load_backbone_from != None:
                    params=torch.load(f"{self.args.client_num_in_total}client_{self.args.load_backbone_from}_{self.args.model}_{self.args.dataset}_{self.args.comm_round}_round_alpha={self.args.partition_alpha}.pth")
                    params={(para[para.find('.')+1:] if para.split('.')[0]!='predictor' else 'linear.'+para[para.find('.')+1:]):params[para] for para in params}
                else:
                    params=torch.load(f"{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.args.comm_round}_round_alpha={self.args.partition_alpha}.pth")
                
                
                self.aggregator.trainer.set_model_params(copy.deepcopy(params))
                for index,client in enumerate(self.client_list):
                    client.trainer.set_model_params(copy.deepcopy(self.aggregator.trainer.get_model_params()))
                self.server_timer.global_comm_round_idx=self.max_comm_round
                wandb.run.summary[f"pretrain"]=False
                # sync the random number generator
                set_seed(self.args.seed)
            except IOError:
                if self.args.load_backbone_from != None:
                    print("ERROR: load_backbone_from but no corresponding backbone!!!!!!!!!")
                    exit()
                max_acc=0
                best_model=copy.deepcopy(self.aggregator.trainer.get_model_params())
                for _ in range(self.max_comm_round):
                    logging.info("################Communication round : {}".format(self.server_timer.global_comm_round_idx))
                    # w_locals = []

                    # Note in the first round, something of global_other_params is not constructed by algorithm_train(),
                    # So care about this.
                    if self.server_timer.global_comm_round_idx == 0:
                        named_params = self.aggregator.get_global_model_params() 
                        if self.args.VHL and self.args.VHL_server_retrain:
                            self.aggregator.server_train_on_noise(max_iterations=50,
                                global_comm_round=0, move_to_gpu=True, dataset_name="Noise Data")
                            named_params = copy.deepcopy(self.aggregator.trainer.get_model_params())

                        params_type = 'model'
                        global_other_params = {}
                        shared_params_for_simulation = {}


                        if self.args.VHL:
                            if self.args.VHL_label_from == "dataset":
                                if self.args.generative_dataset_shared_loader:
                                    shared_params_for_simulation["train_generative_dl_dict"] = self.aggregator.trainer.train_generative_dl_dict
                                    shared_params_for_simulation["test_generative_dl_dict"] = self.aggregator.trainer.test_generative_dl_dict
                                    shared_params_for_simulation["train_generative_ds_dict"] = self.aggregator.trainer.train_generative_ds_dict
                                    shared_params_for_simulation["test_generative_ds_dict"] = self.aggregator.trainer.test_generative_ds_dict
                                    shared_params_for_simulation["noise_dataset_label_shift"] = self.aggregator.trainer.noise_dataset_label_shift
                                    # These two dataloader iters are shared
                                    shared_params_for_simulation["train_generative_iter_dict"] = self.aggregator.trainer.train_generative_iter_dict
                                    shared_params_for_simulation["test_generative_iter_dict"] = self.aggregator.trainer.test_generative_iter_dict
                                else:
                                    global_other_params["train_generative_dl_dict"] = self.aggregator.trainer.train_generative_dl_dict
                                    global_other_params["test_generative_dl_dict"] = self.aggregator.trainer.test_generative_dl_dict
                                    global_other_params["train_generative_ds_dict"] = self.aggregator.trainer.train_generative_ds_dict
                                    global_other_params["test_generative_ds_dict"] = self.aggregator.trainer.test_generative_ds_dict
                                    global_other_params["noise_dataset_label_shift"] = self.aggregator.trainer.noise_dataset_label_shift

                            if self.args.VHL_inter_domain_mapping:
                                global_other_params["VHL_mapping_matrix"] = self.aggregator.trainer.VHL_mapping_matrix

                        if self.args.fed_align:
                            global_other_params["feature_align_means"] = self.aggregator.trainer.feature_align_means


                        if self.args.scaffold:
                            c_global_para = self.aggregator.c_model_global.state_dict()
                            global_other_params["c_model_global"] = c_global_para

                    client_indexes = self.aggregator.client_sampling(
                        self.server_timer.global_comm_round_idx, self.args.client_num_in_total,
                        self.args.client_num_per_round)
                    logging.info("client_indexes = " + str(client_indexes))

                    global_time_info = self.server_timer.get_time_info_to_send()
                    update_state_kargs = self.get_update_state_kargs()

                    named_params, params_type, global_other_params, shared_params_for_simulation = self.algorithm_train(
                        client_indexes,
                        named_params,
                        params_type,
                        global_other_params,
                        update_state_kargs,
                        global_time_info,
                        shared_params_for_simulation
                    )
                    # pick best model
                    acc=self.aggregator.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
                    if acc > max_acc:
                        print("BEST_MODEL_ACC===================", acc)
                        max_acc=acc
                        best_model=copy.deepcopy(self.aggregator.trainer.get_model_params())
                    wandb.log({"training_round":self.server_timer.global_comm_round_idx, "test_acc_in_this_round":acc})
                self.aggregator.trainer.set_model_params(best_model)
                for index,client in enumerate(self.client_list):
                    client.trainer.set_model_params(copy.deepcopy(best_model))
                    # show_memory()
                torch.save(self.aggregator.trainer.get_model_params(), f"{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.args.comm_round}_round_alpha={self.args.partition_alpha}.pth")
                wandb.run.summary[f"pretrain"]=True
                # sync the random number generator
                set_seed(self.args.seed)
                
            
        if self.args.HPFL:
            # print("len(self.aggregator)=================", len(list(self.aggregator.trainer.model.children())))
            # print("self.aggregator=================", list(self.aggregator.trainer.model.children()))
            if self.args.central_trainning_epoch > 0:
                self.train_centrally()
                set_seed(self.args.seed)
            else:
                acc=self.aggregator.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
                print("ACC_ON_FEDAVG_TRAINED::::::::::::::::::::::::", acc)
                wandb.run.summary[f"backbone_acc"] = acc
                set_seed(self.args.seed)
            
            # recover initial freeze_bn to train pluggable without changing bn in backbone
            self.args.freeze_bn=True
            
            # print("before freeze::::::::::::::::", self.client_list[0].trainer.model.state_dict().items())
            # freeze part of model
            if self.args.if_OOD:
                self.OOD_remodel()
                acc=self.aggregator.trainer.OOD_test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
                print("ACC_AFTER_OOD_REMODELED::::::::::::::::::::::::", acc)
                wandb.run.summary[f"backbone_acc"] = acc
                print("before freeze OOD_dete weight::::::::::::::::", self.client_list[0].trainer.OOD_dete.linear.weight)
                print("before freeze OOD_dete bias::::::::::::::::", self.client_list[0].trainer.OOD_dete.linear.bias)
                set_seed(self.args.seed)
            if not self.args.finetune_FedAvg:
                self.freeze()
            else:
                self.args.freeze_bn=False
            if self.server_timer.global_comm_round_idx == self.max_comm_round:
                print("SUCCESSFULLY get the model param")
                named_params = self.aggregator.get_global_model_params() 
                if self.args.VHL and self.args.VHL_server_retrain:
                    self.aggregator.server_train_on_noise(max_iterations=50,
                        global_comm_round=0, move_to_gpu=True, dataset_name="Noise Data")
                    named_params = copy.deepcopy(self.aggregator.trainer.get_model_params())

                params_type = 'model'
                global_other_params = {}
                shared_params_for_simulation = {}

            self.args.global_epochs_per_round=self.args.HPFL_local_iteration
            
            client_indexes = self.aggregator.client_sampling(
                self.server_timer.global_comm_round_idx, self.args.client_num_in_total,
                self.args.client_num_in_total)
            global_time_info = self.server_timer.get_time_info_to_send()
            update_state_kargs = self.get_update_state_kargs()
            
            # train the pluggable
            self.algorithm_train(
                client_indexes,
                named_params,
                params_type,
                global_other_params,
                update_state_kargs,
                global_time_info,
                shared_params_for_simulation
            )
            if self.args.reduce_plug:
                self.aggregator.reduce_plugs()
            # print("after freeze::::::::::::", self.client_list[0].trainer.model.state_dict().items())
            # if self.args.if_OOD:
            #     print("after freeze self.client_list[0].trainer.OOD_dete.linear.weight", self.client_list[0].trainer.OOD_dete.linear.weight)
            #     print("after freeze self.client_list[0].trainer.OOD_dete.linear.bias::::::::::::::::", self.client_list[0].trainer.OOD_dete.linear.bias)
            #     print("after freeze self.client_list[1].trainer.OOD_dete.linear.weight", self.client_list[1].trainer.OOD_dete.linear.weight)
            #     print("after freeze self.client_list[1].trainer.OOD_dete.linear.bias::::::::::::::::", self.client_list[1].trainer.OOD_dete.linear.bias)
            # show_memory()
            # torch.cuda.empty_cache()

        self.total_train_tracker.upload_record_to_wandb()
        self.total_test_tracker.upload_record_to_wandb()


    @abstractmethod
    def algorithm_train(self, client_indexes, named_params, params_type,
                        global_other_params,
                        update_state_kargs, global_time_info,
                        shared_params_for_simulation):
        pass


    # ===========================================================================

    def cfl_train(self):
        # training backbone
        if self.args.comm_round > 0:
            if_freeze="freeze_" if self.args.fcl_freeze_backbone else ""
            try:
                params_before=torch.load(f"fedavg_cfl_{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.args.comm_round//2}_round_alpha={self.args.partition_alpha}.pth")
                params_after=torch.load(f"fedavg_cfl_{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.args.comm_round}_round_alpha={self.args.partition_alpha}.pth")
                if self.args.fcl_freeze_backbone:
                    params_after_freeze=torch.load(f"fedavg_cfl_freeze_{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.args.comm_round}_round_alpha={self.args.partition_alpha}.pth")
                self.aggregator.trainer.set_model_params(copy.deepcopy(params_after))
                self.server_timer.global_comm_round_idx=self.max_comm_round
                for index,client in enumerate(self.client_list):
                    if index < len(self.client_list)//2:
                        client.trainer.set_model_params(copy.deepcopy(params_before))
                    else:
                        client.trainer.set_model_params(copy.deepcopy(params_after))
                self.server_timer.global_comm_round_idx = self.max_comm_round
                # sync the random number generator
                set_seed(self.args.seed)
                print("finishing cfl_training!!!!!")
            except IOError:
                max_acc=0
                best_model=copy.deepcopy(self.aggregator.trainer.get_model_params())
                try:
                    params_before=torch.load(f"fedavg_cfl_{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.args.comm_round//2}_round_alpha={self.args.partition_alpha}.pth")
                    self.aggregator.trainer.set_model_params(copy.deepcopy(params_before))
                    for index,client in enumerate(self.client_list):
                        if index < len(self.client_list)//2:
                            client.trainer.set_model_params(copy.deepcopy(params_before))
                    self.server_timer.global_comm_round_idx=self.max_comm_round//2
                    print("SUCCESSFULLY get the model param")
                    named_params = self.aggregator.get_global_model_params()
                    params_type = 'model'
                    global_other_params = {}
                    shared_params_for_simulation = {}
                    for _ in range(self.max_comm_round//2):
                        logging.info("################Communication round : {}".format(self.server_timer.global_comm_round_idx))

                        if self.server_timer.global_comm_round_idx < self.args.comm_round//2:
                            client_indexes = [0,1,2,3,4]
                        else:
                            client_indexes = [5,6,7,8,9]
                        self.aggregator.selected_clients=client_indexes
                        logging.info("client_indexes = " + str(client_indexes))

                        global_time_info = self.server_timer.get_time_info_to_send()
                        update_state_kargs = self.get_update_state_kargs()

                        # freeze the backbone
                        if self.args.fcl_freeze_backbone and self.server_timer.global_comm_round_idx == self.args.comm_round//2:
                            self.freeze()
                        named_params, params_type, global_other_params, shared_params_for_simulation = self.algorithm_train(
                            client_indexes,
                            named_params,
                            params_type,
                            global_other_params,
                            update_state_kargs,
                            global_time_info,
                            shared_params_for_simulation
                        )
                        # pick best model
                        acc=self.aggregator.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
                        if acc > max_acc:
                            print("BEST_MODEL_ACC===================", acc)
                            max_acc=acc
                            best_model=copy.deepcopy(self.aggregator.trainer.get_model_params())
                        wandb.log({"training_round":self.server_timer.global_comm_round_idx, "test_acc_in_this_round":acc})
                        if self.args.fedavg_cfl and self.server_timer.global_comm_round_idx==(self.args.comm_round//2):
                            torch.save(self.aggregator.trainer.get_model_params(), f"fedavg_cfl_{if_freeze}{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.server_timer.global_comm_round_idx}_round_alpha={self.args.partition_alpha}.pth")
                except IOError:
                    for _ in range(self.max_comm_round):
                        logging.info("################Communication round : {}".format(self.server_timer.global_comm_round_idx))
                        # w_locals = []

                        # Note in the first round, something of global_other_params is not constructed by algorithm_train(),
                        # So care about this.
                        if self.server_timer.global_comm_round_idx == 0:
                            named_params = self.aggregator.get_global_model_params() 
                            if self.args.VHL and self.args.VHL_server_retrain:
                                self.aggregator.server_train_on_noise(max_iterations=50,
                                    global_comm_round=0, move_to_gpu=True, dataset_name="Noise Data")
                                named_params = copy.deepcopy(self.aggregator.trainer.get_model_params())

                            params_type = 'model'
                            global_other_params = {}
                            shared_params_for_simulation = {}

                        if self.server_timer.global_comm_round_idx < self.args.comm_round//2:
                            client_indexes = [0,1,2,3,4]
                        else:
                            client_indexes = [5,6,7,8,9]
                        self.aggregator.selected_clients=client_indexes 
                        logging.info("client_indexes = " + str(client_indexes))

                        global_time_info = self.server_timer.get_time_info_to_send()
                        update_state_kargs = self.get_update_state_kargs()

                        # freeze the backbone & get half_way_model
                        if self.args.fcl_freeze_backbone and self.server_timer.global_comm_round_idx == self.args.comm_round//2:
                            self.freeze()
                            torch.save(self.aggregator.trainer.get_model_params(), f"fedavg_cfl_{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.server_timer.global_comm_round_idx}_round_alpha={self.args.partition_alpha}.pth")
                        named_params, params_type, global_other_params, shared_params_for_simulation = self.algorithm_train(
                            client_indexes,
                            named_params,
                            params_type,
                            global_other_params,
                            update_state_kargs,
                            global_time_info,
                            shared_params_for_simulation
                        )
                        # pick best model
                        acc=self.aggregator.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
                        if acc > max_acc:
                            print("BEST_MODEL_ACC===================", acc)
                            max_acc=acc
                            best_model=copy.deepcopy(self.aggregator.trainer.get_model_params())
                        wandb.log({"training_round":self.server_timer.global_comm_round_idx, "test_acc_in_this_round":acc})
                        if self.args.fedavg_cfl and self.server_timer.global_comm_round_idx==(self.args.comm_round//2):
                                torch.save(self.aggregator.trainer.get_model_params(), f"fedavg_cfl_{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.server_timer.global_comm_round_idx}_round_alpha={self.args.partition_alpha}.pth")
                if self.args.fedavg_cfl:
                    torch.save(self.aggregator.trainer.get_model_params(), f"fedavg_cfl_{if_freeze}{self.args.client_num_in_total}client_{self.args.algorithm}_{self.args.model}_{self.args.dataset}_{self.args.comm_round}_round_alpha={self.args.partition_alpha}.pth")
                
                self.aggregator.trainer.set_model_params(best_model)
                for index,client in enumerate(self.client_list):
                    client.trainer.set_model_params(copy.deepcopy(best_model))
                wandb.run.summary[f"pretrain"]=True
                # sync the random number generator
                set_seed(self.args.seed)
        # training plug-ins
        if self.args.HPFL:
            self.aggregator.trainer.set_model_params(copy.deepcopy(params_before))
            self.server_timer.global_comm_round_idx = self.max_comm_round
            # recover initial freeze_bn to train pluggable without changing bn in backbone
            self.args.freeze_bn=True
            
            # freeze part of model
            if not self.args.finetune_FedAvg:
                self.freeze()
            else:
                self.args.freeze_bn=False
            if self.server_timer.global_comm_round_idx == self.max_comm_round:
                print("SUCCESSFULLY get the model param")
                named_params = self.aggregator.get_global_model_params()
                params_type = 'model'
                global_other_params = {}
                shared_params_for_simulation = {}

            self.args.global_epochs_per_round=self.args.HPFL_local_iteration
            
            client_indexes = [0,1,2,3,4]
            self.aggregator.selected_clients = client_indexes
            global_time_info = self.server_timer.get_time_info_to_send()
            update_state_kargs = self.get_update_state_kargs()
            
            # train the pluggable
            self.algorithm_train(
                client_indexes,
                named_params,
                params_type,
                global_other_params,
                update_state_kargs,
                global_time_info,
                shared_params_for_simulation
            )
            
            # training plug-ins on 5-9 clients
            self.aggregator.trainer.set_model_params(copy.deepcopy(params_before))
            self.server_timer.global_comm_round_idx = self.max_comm_round
            # recover initial freeze_bn to train pluggable without changing bn in backbone
            self.args.freeze_bn=True
            
            
            
            # freeze part of model
            if not self.args.finetune_FedAvg:
                self.freeze()
            else:
                self.args.freeze_bn=False
            
            # test the backbone accuracy, MUST AFTER FREEZE, OR BACKBONE WILL CHANGE?
            # acc=self.aggregator.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
            # print("ACC_ON_FEDAVG_TRAINED::::::::::::::::::::::::", acc)
            # wandb.run.summary[f"backbone_acc"] = acc
            
            if self.server_timer.global_comm_round_idx == self.max_comm_round:
                print("SUCCESSFULLY get the model param")
                named_params = self.aggregator.get_global_model_params() 
                params_type = 'model'
                global_other_params = {}
                shared_params_for_simulation = {}

            self.args.global_epochs_per_round=self.args.HPFL_local_iteration
            
            client_indexes = [5,6,7,8,9]
            self.aggregator.selected_clients = client_indexes
            global_time_info = self.server_timer.get_time_info_to_send()
            update_state_kargs = self.get_update_state_kargs()
            
            # train the pluggable
            self.algorithm_train(
                client_indexes,
                named_params,
                params_type,
                global_other_params,
                update_state_kargs,
                global_time_info,
                shared_params_for_simulation
            )
            
            # # training plug-ins on 0-9 clients for Naive FCL
            # self.aggregator.trainer.set_model_params(copy.deepcopy(params_after))
            # self.server_timer.global_comm_round_idx = self.max_comm_round
            # # recover initial freeze_bn to train pluggable without changing bn in backbone
            # self.args.freeze_bn=True
            
            # # freeze part of model
            # if not self.args.finetune_FedAvg:
            #     self.freeze()
            # else:
            #     self.args.freeze_bn=False
            # if self.server_timer.global_comm_round_idx == self.max_comm_round:
            #     print("SUCCESSFULLY get the model param")
            #     named_params = self.aggregator.get_global_model_params() 
            #     params_type = 'model'
            #     global_other_params = {}
            #     shared_params_for_simulation = {}
            
            # self.args.global_epochs_per_round=self.args.HPFL_local_iteration
            
            # client_indexes = [0,1,2,3,4,5,6,7,8,9]
            # self.aggregator.selected_clients = client_indexes
            # global_time_info = self.server_timer.get_time_info_to_send()
            # update_state_kargs = self.get_update_state_kargs()
            
            # # train the pluggable
            # self.algorithm_train(
            #     client_indexes,
            #     named_params,
            #     params_type,
            #     global_other_params,
            #     update_state_kargs,
            #     global_time_info,
            #     shared_params_for_simulation
            # )


    def cfl_test(self):
        ACC_list=[]
        for idx, client in enumerate(self.client_list):
            acc=client.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
            ACC_list.append(acc)
        ACC_list=np.array(ACC_list)
        acc=np.mean(ACC_list)
        var=np.var(ACC_list)
        
        print(f"AVG_GLOBAL_ACC_ON_PERSONALIZED_MODEL",acc)
        print(f"GLOBAL_ACC_ON_PERSONALIZED_MODEL",ACC_list)
        
        wandb.run.summary[f"AVG_GLOBAL_ACC_ON_PERSONALIZED_MODEL"]=acc
        wandb.run.summary[f"GLOBAL_ACC_ON_PERSONALIZED_MODEL"]=ACC_list
        wandb.run.summary[f"VAR_GLOBAL_ACC_ON_PERSONALIZED_MODEL"]=var

        pass
