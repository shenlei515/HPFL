import copy
import torch

from .minimizers import SAM
from algorithms.fedavg.fedavg_client_timer import FedAVGClientTimer

import logging
import copy

import numpy as np
from sklearn import feature_extraction
import torch

from algorithms_standalone.basePS.client import Client

from algorithms.fedavg.fedavg_client_timer import FedAVGClientTimer

from utils.data_utils import (
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations,
    get_num_iterations,
)

from utils.model_utils import (
    set_freeze_by_names,
    get_actual_layer_names,
    freeze_by_names,
    unfreeze_by_names,
    get_modules_by_names
)


from utils.checkpoint import save_images

from model.build import create_model


class FedSAMClient(Client):
        
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                train_data_num, device, args, model_trainer, perf_timer=None, metrics=None):
        super().__init__(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                train_data_num, device, args, model_trainer, perf_timer, metrics)
        local_num_iterations_dict = {}
        local_num_iterations_dict[self.client_index] = self.local_num_iterations

        self.global_epochs_per_round = self.args.global_epochs_per_round
        local_num_epochs_per_comm_round_dict = {}
        local_num_epochs_per_comm_round_dict[self.client_index] = self.args.global_epochs_per_round

        # override the PSClientManager's timer
        self.client_timer = FedAVGClientTimer(
            self.args,
            self.local_num_iterations,
            local_num_iterations_dict,
            self.global_epochs_per_round,
            local_num_epochs_per_comm_round_dict,
            client_index=self.client_index 
        )
        self.train_tracker.timer = self.client_timer
        self.test_tracker.timer = self.client_timer

        # ===========================================================
        if self.args.scaffold:
            self.c_model_local = create_model(self.args,
                model_name=self.args.model, output_dim=self.args.model_output_dim)
            for name, params in self.c_model_local.named_parameters():
                params.data = params.data*0
        
    # override
    def lr_schedule(self, num_iterations, warmup_epochs):
        epoch = None
        iteration = None
        round_idx = self.client_timer.local_comm_round_idx 
        if self.args.sched == "no":
            pass
        else:
            if round_idx < warmup_epochs:
                # Because gradual warmup need iterations updates
                self.trainer.warmup_lr_schedule(round_idx*num_iterations)
            else:
                self.trainer.lr_schedule(round_idx)

    def get_max_comm_round(self):
        # return self.args.max_comm_round + 1
        return self.args.max_epochs // self.args.global_epochs_per_round + 1


    def get_current_num_iterations(self, local_epoch=0):
        if self.args.fedavg_local_step_type == "whole":
            local_num_iterations = None
        elif self.args.fedavg_local_step_type == "fixed":
            local_num_iterations = get_num_iterations(
                self.train_data_local_num_dict,
                batch_size=self.args.batch_size,
                type="default",
                default=10
            )
        elif self.args.fedavg_local_step_type == "fixed2whole":
            # global_comm_round = self.client_timer.global_comm_round_idx
            global_outer_epoch_idx = self.client_timer.global_outer_epoch_idx

            time_coefficient = global_outer_epoch_idx / self.args.max_epochs
            fixed_local_num_iterations = get_num_iterations(
                self.train_data_local_num_dict,
                batch_size=self.args.batch_size,
                type="default",
                default=10
            )
            whole_local_num_iterations = self.local_sample_number // self.args.batch_size
            local_num_iterations = int(fixed_local_num_iterations*(1 - time_coefficient) \
                + whole_local_num_iterations*time_coefficient)
        else:
            raise NotImplementedError
        return local_num_iterations

    def run_epoch_with_mixup(self, optimizer, criterion):
        minimizer = SAM(optimizer, self.model, self.rho, self.eta)
        running_loss = 0.0
        i = 0
        for inputs, targets in self.trainloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            inputs, targets_a, targets_b, lam = self.mixup_data(inputs, targets)

            # Ascent Step
            outputs = self.model(inputs)
            loss = self.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            minimizer.ascent_step()

            # Descent Step
            loss = self.mixup_criterion(criterion, self.model(inputs), targets_a, targets_b, lam)
            loss.backward()
            minimizer.descent_step()

            with torch.no_grad():
                running_loss += loss.item()

            i += 1
        if i == 0:
            print("Not running epoch", self.id)
            return 0
        return running_loss / i
    
    def fedavg_train(self, round_idx=None, global_other_params=None, tracker=None, metrics=None,
                shared_params_for_simulation=None,
                    **kwargs):

        client_other_params = {}
        train_epoch_kwargs = {}
        train_epoch_kwargs["global_outer_epoch_idx"] = self.client_timer.global_outer_epoch_idx

        if self.args.if_get_diff or self.args.model_dif_track or self.args.fedprox or self.args.scaffold:
            previous_model = copy.deepcopy(self.trainer.get_model_params())
            # kwargs["previous_model"] = previous_model
            train_epoch_kwargs["previous_model"] = previous_model

        if self.args.scaffold:
            c_model_global = global_other_params["c_model_global"]
            # for name, param in c_model_global.items():
            #     param.data = param.data.to(self.device)
            for name in c_model_global:
                c_model_global[name] = c_model_global[name].to(self.device)
            self.c_model_local.to(self.device)
            c_model_local = self.c_model_local.state_dict()

            train_epoch_kwargs['c_model_global'] = c_model_global
            train_epoch_kwargs['c_model_local'] = c_model_local

        iteration_cnt = 0
        # local clinet training process(multi-epoch)
        for epoch in range(self.args.global_epochs_per_round):
            self.epoch_init()

            things_to_track = []
            
            current_num_iterations = self.get_current_num_iterations(epoch)
            self.trainer.FedSAM_train_one_epoch(
                self.train_local, self.device, self.args,
                epoch, tracker, metrics,
                local_iterations=current_num_iterations,
                checkpoint_extra_name=f"client{self.client_index}",
                things_to_track=things_to_track,
                **train_epoch_kwargs)

        if self.args.if_get_diff:
            compressed_weights_diff, model_indexes = self.get_model_diff_params(previous_model)
        else:
            compressed_weights_diff, model_indexes = self.get_model_params()

        if self.args.model_dif_track:
            weights = self.trainer.get_model_params()
            tracker.update_local_record(
                'model_dif_track',
                self.client_index, 
                summary_n_samples=self.local_num_iterations*self.args.global_epochs_per_round,
                args=self.args,
                choose_layers=True,
                track_thing='model_dif_epoch_track',
                weights_1=weights,
                weights_2=previous_model
            )
        if self.args.tSNE_track:
            data_tsne, labels = self.trainer.feature_reduce(
                time_stamp=self.client_timer.global_comm_round_idx,
                reduce_method="tSNE",
                extra_name="FL", postfix=f"Cli{self.client_index}",
                batch_data=None, data_loader=self.test_local, num_points=1000)

        return compressed_weights_diff, model_indexes, self.local_sample_number, client_other_params, shared_params_for_simulation
    
    def algorithm_on_train(self, update_state_kargs, 
        client_index, named_params, params_type='model',
        global_other_params=None,
        traininig_start=False,
        shared_params_for_simulation=None, **kwargs):

        client_other_params = {}
        self.lr_schedule(self.global_num_iterations, self.args.warmup_epochs)
        # logging.info(f"End lr schedulering::: !!!!  self.client_timer.local_comm_round_idx:\
        #     {self.client_timer.local_comm_round_idx} \
        #     lr: {self.trainer.lr_scheduler.lr} \
        #     self.trainer.lr_scheduler.optimizer.param_groups[0]['lr']: {self.trainer.lr_scheduler.optimizer.param_groups[0]['lr']} \n\n")

        if params_type == 'model':
            self.set_model_params(named_params)
        elif params_type == 'grad':
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.args.VHL:
            if self.args.VHL_label_from == "dataset":
                if self.args.generative_dataset_shared_loader:
                    self.trainer.train_generative_dl_dict = shared_params_for_simulation["train_generative_dl_dict"]
                    self.trainer.test_generative_dl_dict = shared_params_for_simulation["test_generative_dl_dict"]
                    self.trainer.train_generative_ds_dict = shared_params_for_simulation["train_generative_ds_dict"]
                    self.trainer.test_generative_ds_dict = shared_params_for_simulation["test_generative_ds_dict"]
                    self.trainer.noise_dataset_label_shift = shared_params_for_simulation["noise_dataset_label_shift"]
                    # These two dataloader iters are shared
                    self.trainer.train_generative_iter_dict = shared_params_for_simulation["train_generative_iter_dict"]
                    self.trainer.test_generative_iter_dict = shared_params_for_simulation["test_generative_iter_dict"]
                else:
                    self.trainer.train_generative_dl_dict = global_other_params["train_generative_dl_dict"]
                    self.trainer.test_generative_dl_dict = global_other_params["test_generative_dl_dict"]
                    self.trainer.train_generative_ds_dict = global_other_params["train_generative_ds_dict"]
                    self.trainer.test_generative_ds_dict = global_other_params["test_generative_ds_dict"]
                    self.trainer.noise_dataset_label_shift = global_other_params["noise_dataset_label_shift"]


            if self.args.VHL_inter_domain_mapping:
                self.trainer.set_VHL_mapping_matrix(global_other_params["VHL_mapping_matrix"])

        if self.args.fed_align:
            self.trainer.set_feature_align_means(global_other_params["feature_align_means"])
        model_params, model_indexes, local_sample_number, client_other_params, shared_params_for_simulation = \
            self.fedavg_train(
                self.client_timer.global_comm_round_idx,
                global_other_params,
                self.train_tracker,
                self.metrics,
                shared_params_for_simulation, **kwargs)

        if self.args.record_dataframe:
            """
                Now, the client timer need to be updated by each iteration, 
                so as to record the track infomation.
            """
            pass
        else:
            if traininig_start:
                self.client_timer.past_epochs(epochs=self.global_epochs_per_round-1)
            else:
                self.client_timer.past_epochs(epochs=self.global_epochs_per_round)
        return model_params, model_indexes, local_sample_number, client_other_params, shared_params_for_simulation