import copy
import logging
import random
from re import M
import numpy as np
import torch
import wandb

from .client import FedSAMClient
from .aggregator import FedSAMAggregator

from utils.data_utils import (
    get_avg_num_iterations,
    get_label_distribution,
    get_selected_clients_label_distribution
)
from utils.checkpoint import setup_checkpoint_config, save_checkpoint


from algorithms_standalone.basePS.basePSmanager import BasePSManager
from model.build import create_model
from trainers.build import create_trainer

from algorithms.fedavg.fedavg_server_timer import FedAVGServerTimer

class FEDSAMManager(BasePSManager):
    def __init__(self, device, args):
        super().__init__(device, args)
        del self.client_list, self.aggregator
        self.client_list = []

        self.global_epochs_per_round = self.args.global_epochs_per_round
        local_num_epochs_per_comm_round_dict = {}
        # local_num_epochs_per_comm_round_dict[self.client_index] = self.args.global_epochs_per_round

        self.server_timer = FedAVGServerTimer(
            self.args,
            self.global_num_iterations,
            None,
            self.global_epochs_per_round,
            local_num_epochs_per_comm_round_dict
        )
        self.total_train_tracker.timer = self.server_timer
        self.total_test_tracker.timer = self.server_timer
        
        self.args.client_optimizer = "SAM"
        self._setup_clients()
        self._setup_server()



    def _setup_server(self):
        logging.info("############_setup_server (START)#############")
        model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                            device=self.device, pretrained=self.args.pretrained, **self.other_params)
        num_iterations = get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)
        init_state_kargs = self.get_init_state_kargs()
        model_trainer = create_trainer(
            self.args, self.device, model, num_iterations=num_iterations,
            train_data_num=self.train_data_num_in_total, test_data_num=self.test_data_num_in_total,
            train_data_global=self.train_global, test_data_global=self.test_global,
            train_data_local_num_dict=self.train_data_local_num_dict, train_data_local_dict=self.train_data_local_dict,
            test_data_local_dict=self.test_data_local_dict, class_num=self.class_num, other_params=self.other_params,
            server_index=0, role='server',
            **init_state_kargs
        )
        # model_trainer = create_trainer(self.args, self.device, model)
        self.aggregator = FedSAMAggregator(self.train_global, self.test_global, self.train_data_num_in_total,
                self.train_data_local_dict, self.test_data_local_dict,
                self.train_data_local_num_dict, self.args.client_num_in_total, self.device,
                self.args, model_trainer, perf_timer=self.perf_timer, metrics=self.metrics, traindata_cls_counts=self.traindata_cls_counts)

        # self.aggregator.traindata_cls_counts = self.traindata_cls_counts
        logging.info("############_setup_server (END)#############")

    def _setup_clients(self):
        logging.info("############setup_clients (START)#############")
        init_state_kargs = self.get_init_state_kargs()
        # for client_index in range(self.args.client_num_in_total):
        for client_index in range(self.number_instantiated_client):
            model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                            device=self.device, pretrained=self.args.pretrained, **self.other_params)
            num_iterations = get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)
            model_trainer = create_trainer(
                self.args, self.device, model, num_iterations=num_iterations,
                train_data_num=self.train_data_num_in_total, test_data_num=self.test_data_num_in_total,
                train_data_global=self.train_global, test_data_global=self.test_global,
                train_data_local_num_dict=self.train_data_local_num_dict, train_data_local_dict=self.train_data_local_dict,
                test_data_local_dict=self.test_data_local_dict, class_num=self.class_num, other_params=self.other_params,
                client_index=client_index, role='client',
                **init_state_kargs
            )
            # model_trainer = create_trainer(self.args, self.device, model)
            c = FedSAMClient(client_index, self.train_data_local_dict, self.train_data_local_num_dict, 
                    self.test_data_local_dict, self.train_data_num_in_total,
                    self.device, self.args, model_trainer,
                    perf_timer=self.perf_timer, metrics=self.metrics)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")



    # override
    def check_end_epoch(self):
        return True

    # override
    def check_test_frequency(self):
        return ( self.server_timer.global_comm_round_idx % self.args.frequency_of_the_test == 0 \
            or self.server_timer.global_comm_round_idx == self.max_comm_round - 1)

    # override
    def check_and_test(self):
        if self.check_test_frequency():
            self.test()
        else:
            self.reset_train_test_tracker()

    def algorithm_train(self, client_indexes, named_params, params_type,
                        global_other_params,
                        update_state_kargs, global_time_info,
                        shared_params_for_simulation):
        for i, client_index in enumerate(client_indexes):

            copy_global_other_params = copy.deepcopy(global_other_params)
            if self.args.exchange_model == True:
                copy_named_model_params = copy.deepcopy(named_params)

            if self.args.instantiate_all:
                client = self.client_list[client_index]
            else:
                # WARNING! All instantiated clients are only used in current round.
                # The history information saved may cause BUGs in the realistic FL scenario.
                client = self.client_list[i]

            if global_time_info["global_time_info"]["global_comm_round_idx"] == 0:
                traininig_start = True
            else:
                traininig_start = False

            # client training.............
            model_params, model_indexes, local_sample_number, client_other_params, \
            local_train_tracker_info, local_time_info, shared_params_for_simulation = \
                    client.train(update_state_kargs, global_time_info, 
                    client_index, copy_named_model_params, params_type,
                    copy_global_other_params,
                    traininig_start=traininig_start,
                    shared_params_for_simulation=shared_params_for_simulation)

            self.total_train_tracker.decode_local_info(client_index, local_train_tracker_info)
            # self.total_test_tracker.decode_local_info(client_index, local_test_tracker_info)

            self.server_timer.update_time_info(local_time_info)
            self.aggregator.add_local_trained_result(
                client_index, model_params, model_indexes, local_sample_number, client_other_params)

            if self.server_timer.global_comm_round_idx==self.max_comm_round:
                # upload the Pluggable,and save it in aggregator
                print("upload the Pluggable,and save it in aggregator")
                self.aggregator.model_list.append(copy.deepcopy(list(client.trainer.model.children())[-self.args.freeze_layer:]))
                # upload the distribution of feature extracted with backbone
                # print("client.cal_dist(client_index)",client.cal_dist(client_index))
                print("upload successfully")
                
            # show_memory()
            # torch.cuda.empty_cache()

        # update global weights and return them
        # global_model_params = self.aggregator.aggregate()
        if self.server_timer.global_comm_round_idx!=self.max_comm_round:
            global_model_params, global_other_params, shared_params_for_simulation = self.aggregator.aggregate(
                global_comm_round=self.server_timer.global_comm_round_idx,
                global_outer_epoch_idx=self.server_timer.global_outer_epoch_idx,
                tracker=self.total_train_tracker,
                metrics=self.metrics)

        params_type = 'model'
        
        self.check_and_test()

        save_checkpoints_config = setup_checkpoint_config(self.args) if self.args.checkpoint_save else None
        save_checkpoint(
            self.args, save_checkpoints_config, extra_name="server", 
            epoch=self.server_timer.global_outer_epoch_idx,
            model_state_dict=self.aggregator.get_global_model_params(),
            optimizer_state_dict=self.aggregator.trainer.optimizer.state_dict(),
            train_metric_info=self.total_train_tracker.get_metric_info(self.metrics),
            test_metric_info=self.total_test_tracker.get_metric_info(self.metrics),
            postfix=self.args.checkpoint_custom_name
        )
        self.server_timer.past_epochs(epochs=1*self.global_epochs_per_round)
        self.server_timer.past_comm_round(comm_round=1)
        
        if self.server_timer.global_comm_round_idx!=self.max_comm_round+1:
            return global_model_params, params_type, global_other_params, shared_params_for_simulation
        else:
            return
        
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
                
        acc=self.aggregator.trainer.test(self.test_global, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)
        print("ACC_ON_FEDAVG_TRAINED::::::::::::::::::::::::", acc)
        wandb.run.summary[f"backbone_acc"] = acc

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
        
    def test_with_personalized_batch(self, select_index="index"):
        logging.info("################test_with_personalized_batch:")
        acc=0
        num_sample_a=0
        for idx,client in enumerate(self.client_list):
            test_data=[self.p_loader[i] for i in self.personalized_map[idx]]
            num_sample_c=len(test_data)
            X, y=list(zip(*test_data))
            dataset=torch.utils.data.TensorDataset(torch.cat(X),torch.cat(y))
            test_data=torch.utils.data.dataloader.DataLoader(dataset, batch_size=128)
            num_sample_a+=num_sample_c
            if select_index=="personalized_with_OOD":
                acc+=client.trainer.OOD_test(test_data, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)*num_sample_c
            else:
                acc+=client.trainer.test(test_data, device=self.device, tracker=self.total_test_tracker, metrics=self.metrics)*num_sample_c
        acc/=num_sample_a*1.0
        print(f"ACC_FOR_PERSONALIZED_BATCH_WITH{select_index}:::::::::::::::::::::::::::::::::::::::",acc)
        wandb.run.summary[f"personalized_acc_with_{select_index}"] = acc
    
    def test_plug_globally(self):
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




