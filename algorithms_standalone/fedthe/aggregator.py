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
from utils.corr_methods_CKA import cka, gram_linear, gram_rbf
from algorithms_standalone.fedthe.client import LocalModel

class FedTHEAggregator(Aggregator):
    def __init__(self, train_global, test_global, all_train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                args, model_trainer, perf_timer=None, metrics=None, traindata_cls_counts=None):
        super().__init__(train_global, test_global, all_train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                args, model_trainer, perf_timer, metrics, traindata_cls_counts=traindata_cls_counts)
        self.feat_list=[]

        pred=copy.deepcopy(self.trainer.model.linear)
        self.trainer.model.linear=torch.nn.Identity()
        self.trainer.model=LocalModel(self.trainer.model, pred)

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