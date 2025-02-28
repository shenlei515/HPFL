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
    check_type,
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
from algorithms_standalone.fedrep.client import LocalModel

class FEDREPAggregator(Aggregator):
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
        
        pred=copy.deepcopy(self.trainer.model.linear)
        self.trainer.model.linear=torch.nn.Identity()
        self.trainer.model=LocalModel(self.trainer.model, pred)

    def get_max_comm_round(self):
        # if self.args.HPFL:
            # return 1
        return self.args.comm_round
    
    def aggregate(self, global_comm_round=0, global_outer_epoch_idx=0, tracker=None, metrics=None,
                ):
        start_time = time.time()
        model_list = []
        training_num = 0

        global_other_params = {}
        shared_params_for_simulation = {}

        if self.args.model_dif_track:
            previous_model = copy.deepcopy(self.get_global_model_params())

        if self.args.if_get_diff is True and self.args.psgd_exchange == "model":
            logging.debug("Server is averaging model diff!!")
            averaged_params = self.get_global_model_params()
            # for idx in range(self.worker_num):
            for idx in self.selected_clients:
                model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
                training_num += self.sample_num_dict[idx]

            logging.debug("len of self.model_dict[idx] = " + str(len(self.model_dict)))
            # aggregate model(average)
            for k in averaged_params.keys():
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    # logging.info("averaged_params[k].dtype: {}, local_model_params[k].dtype: {}".format(
                    #     averaged_params[k].dtype, local_model_params[k].dtype
                    # ))
                    averaged_params[k] += (local_model_params[k] * w).type(averaged_params[k].dtype)
        elif self.args.if_get_diff is False:
            logging.debug("Server is averaging model or adding grads!!")
            # for idx in range(self.worker_num):
            sample_num_list = []
            client_other_params_list = []
            for idx in self.selected_clients:
                model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
                sample_num_list.append(self.sample_num_dict[idx])
                if idx in self.client_other_params_dict:
                    client_other_params = self.client_other_params_dict[idx]
                else:
                    client_other_params = {}
                client_other_params_list.append(client_other_params)
                training_num += self.sample_num_dict[idx]

            logging.debug("len of self.model_dict[idx] = " + str(len(self.model_dict)))
            logging.info("Aggregator: using average type: {} ".format(
                self.args.fedavg_avg_weight_type
            ))

            average_weights_dict_list, homo_weights_list = self.get_average_weight_dict(
                sample_num_list=sample_num_list,
                client_other_params_list=client_other_params_list,
                global_comm_round=global_comm_round,
                global_outer_epoch_idx=global_outer_epoch_idx)

            averaged_params = average_named_params(
                model_list,
                average_weights_dict_list
            )

            if self.args.fed_align:
                global_other_params["feature_align_means"] = self.trainer.feature_align_means
            if self.args.scaffold:
                c_delta_para_list = []
                for i, client_other_params in enumerate(client_other_params_list):
                    c_delta_para_list.append(client_other_params["c_delta_para"])

                total_delta = copy.deepcopy(c_delta_para_list[0])
                # for key, param in total_delta.items():
                #     param.data = 0.0
                for key in total_delta:
                    total_delta[key] = 0.0

                for c_delta_para in c_delta_para_list:
                    for key, param in total_delta.items():
                        total_delta[key] += c_delta_para[key] / len(client_other_params_list)

                c_global_para = self.c_model_global.state_dict()
                for key in c_global_para:
                    # logging.debug(f"total_delta[key].device : {total_delta[key].device}, \
                    # c_global_para[key].device : {c_global_para[key].device}")

                    c_global_para[key] += check_type(total_delta[key], c_global_para[key].type())
                self.c_model_global.load_state_dict(c_global_para)
                global_other_params["c_model_global"] = c_global_para

        else:
            raise NotImplementedError

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)


        if self.args.tSNE_track:
            data_tsne, labels = self.trainer.feature_reduce(
                time_stamp=global_comm_round,
                reduce_method="tSNE",
                extra_name="FedAvg", postfix="server",
                batch_data=None, data_loader=self.test_global, num_points=1000)

        if self.args.model_dif_track:
            global_model_weights = self.trainer.get_model_params()
            if self.args.model_layer_dif_divergence_track:
                global_named_modules = self.trainer.get_model_named_modules()
                tracker.update_local_record(
                    'model_dif_track',
                    server_index=0, 
                    summary_n_samples=self.global_num_iterations*1,
                    args=self.args,
                    choose_layers=True,
                    track_thing='model_layer_dif_divergence_track',
                    global_model_weights=global_model_weights,
                    model_list=model_list,
                    selected_clients=self.selected_clients,
                    global_named_modules=global_named_modules,
                )
            if self.args.model_dif_divergence_track:
                tracker.update_local_record(
                    'model_dif_track',
                    server_index=0, 
                    summary_n_samples=self.global_num_iterations*1,
                    args=self.args,
                    choose_layers=True,
                    track_thing='model_dif_divergence_track',
                    global_model_weights=global_model_weights,
                    model_list=model_list,
                    selected_clients=self.selected_clients,
                )
        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))

        averaged_params=averaged_params
        return averaged_params, global_other_params, shared_params_for_simulation

    def select_pluggable(self,test_data,select_index, client_list=None):
        if select_index=="index":
            return np.array(range(len(self.dist_list))).astype(int)
        else:
            assert False
        
    # def set_global_model_params(self, weights):
    #     for name, layer in list(self.trainer.model.base.named_children()):
    #         for param_name, _ in layer.state_dict().items():
    #             print(param_name)
    #             layer.state_dict()[param_name].copy_(weights['base.'+name+"."+param_name])

