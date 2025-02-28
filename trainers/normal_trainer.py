import copy
import logging
import time

import torch
import wandb
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from torch.distributions import Categorical

from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from fedml_core.trainer.model_trainer import ModelTrainer

from data_preprocessing.utils.stats import record_batch_data_stats

from utils.data_utils import (
    get_data,
    get_named_data,
    get_all_bn_params,
    apply_gradient,
    clear_grad,
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations,
    check_device,
    get_train_batch_data
)

from utils.model_utils import (
    set_freeze_by_names,
    get_actual_layer_names,
    freeze_by_names,
    unfreeze_by_names,
    get_modules_by_names
)

from utils.matrix_utils import orthogo_tensor

from utils.distribution_utils import train_distribution_diversity

from utils.context import (
    raise_error_without_process,
    get_lock,
)

from utils.checkpoint import (
    setup_checkpoint_config, setup_save_checkpoint_path, save_checkpoint,
    setup_checkpoint_file_name_prefix,
    save_checkpoint_without_check
)


from model.build import create_model
from data_preprocessing.build import (
    VHL_load_dataset
)

from trainers.averager import Averager
from trainers.tSNE import Dim_Reducer

from loss_fn.cov_loss import (
    cov_non_diag_norm, cov_norm
)
from loss_fn.losses import LabelSmoothingCrossEntropy, proxy_align_loss, align_feature_loss
# from memory_profiler import profile


class NormalTrainer(ModelTrainer):
    def __init__(self, model, device, criterion, optimizer, lr_scheduler, args, **kwargs):
        super().__init__(model)

        if kwargs['role'] == 'server':
            if "server_index" in kwargs:
                self.server_index = kwargs["server_index"]
            else:
                self.server_index = args.server_index
            self.client_index = None
            self.index = self.server_index

        elif kwargs['role'] == 'client':
            if "client_index" in kwargs:
                self.client_index = kwargs["client_index"]
            else:
                self.client_index = args.client_index
            self.server_index = None
            self.index = self.client_index
        else:
            raise NotImplementedError

        self.role = kwargs['role']

        self.args = args
        self.model = model
        # self.model.to(device)
        self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer

        self.save_checkpoints_config = setup_checkpoint_config(self.args)

        # For future use
        if args.trainer_param_groups:
            self.param_groups = self.optimizer.param_groups
            with raise_error_without_process():
                self.param_names = list(
                    enumerate([group["name"] for group in self.param_groups])
                )

        self.named_parameters = list(self.model.named_parameters())

        if len(self.named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                    in sorted(self.named_parameters)}
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {v: 'noname.%s' % i
                                    for param_group in self.param_groups
                                    for i, v in enumerate(param_group['params'])}

        self.averager = Averager(self.args, self.model)

        self.lr_scheduler = lr_scheduler


        if self.args.VHL:
            if self.args.VHL_feat_align:
                if self.args.VHL_inter_domain_ortho_mapping:
                    self.VHL_mapping_matrix = None
                else:
                    self.VHL_mapping_matrix = torch.rand(
                        self.args.model_feature_dim, self.args.model_feature_dim)
                self.proxy_align_loss = proxy_align_loss(
                    inter_domain_mapping=self.args.VHL_inter_domain_mapping,
                    inter_domain_class_match=self.args.VHL_class_match,
                    noise_feat_detach=self.args.VHL_feat_detach,
                    noise_contrastive=self.args.VHL_noise_contrastive,
                    inter_domain_mapping_matrix=self.VHL_mapping_matrix,
                    inter_domain_weight=self.args.VHL_feat_align_inter_domain_weight,
                    inter_class_weight=self.args.VHL_feat_align_inter_cls_weight,
                    noise_supcon_weight=self.args.VHL_noise_supcon_weight,
                    noise_label_shift=self.args.num_classes,
                    device=self.device)
            if self.args.VHL_data == 'dataset':
                if self.args.VHL_dataset_from_server:
                    self.train_generative_dl_dict = {}
                    self.test_generative_dl_dict = {}
                    self.train_generative_ds_dict = {}
                    self.test_generative_ds_dict = {}
                    self.dataset_label_shift = {}

                    self.train_generative_iter_dict = {}
                    self.test_generative_iter_dict = {}

                else:
                    self.create_noise_dataset_dict()
            else:
                raise NotImplementedError

        if self.args.fed_align:
            self.feature_align_means = torch.rand(
                self.args.num_classes, self.args.model_feature_dim
            )
            self.align_feature_loss = align_feature_loss(
                self.feature_align_means, self.args.fed_align_std, device
            )
    def track(self, tracker, summary_n_samples, model, loss, end_of_epoch,
            checkpoint_extra_name="centralized",
            things_to_track=[]):
        pass

    def epoch_init(self):
        pass

    def epoch_end(self):
        pass

    def update_state(self, **kwargs):
        # This should be called begin the training of each epoch.
        self.update_loss_state(**kwargs)
        self.update_optimizer_state(**kwargs)

    def update_loss_state(self, **kwargs):
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss"]:
            kwargs['cls_num_list'] = kwargs["selected_cls_num_list"]
            self.criterion.update(**kwargs)
        elif self.args.loss_fn in ["local_FocalLoss", "local_LDAMLoss"]:
            kwargs['cls_num_list'] = kwargs["local_cls_num_list_dict"][self.index]
            self.criterion.update(**kwargs)

    def update_optimizer_state(self, **kwargs):
        pass


    def generate_fake_data(self, num_of_samples=64):
        input = torch.randn(num_of_samples, self.args.model_input_channels,
                    self.args.dataset_load_image_size, self.args.dataset_load_image_size)
        return input


    def get_model_named_modules(self):
        return dict(self.model.cpu().named_modules())


    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        # for name, param in model_parameters.items():
        #     logging.info(f"Getting params as model_parameters: name:{name}, shape: {param.shape}")
        if self.args.load_backbone_from == None:
            self.model.load_state_dict(model_parameters, strict=True)
        else: # FedRod + HPFL
            self.model.load_state_dict(model_parameters, strict=True)


    def set_VHL_mapping_matrix(self, VHL_mapping_matrix):
        self.VHL_mapping_matrix = VHL_mapping_matrix
        self.proxy_align_loss.inter_domain_mapping_matrix = VHL_mapping_matrix

    def get_VHL_mapping_matrix(self):
        return self.VHL_mapping_matrix


    def set_feature_align_means(self, feature_align_means):
        self.feature_align_means = feature_align_means
        self.align_feature_loss.feature_align_means = feature_align_means

    def get_feature_align_means(self):
        return self.feature_align_means
    def create_noise_dataset_dict(self):
        self.train_generative_dl_dict, self.test_generative_dl_dict, \
        self.train_generative_ds_dict, self.test_generative_ds_dict \
            = VHL_load_dataset(self.args)
        self.noise_dataset_label_shift = {}
        noise_dataset_label_init = 0
        next_label_shift = noise_dataset_label_init
        for dataset_name in self.train_generative_dl_dict.keys():
            self.noise_dataset_label_shift[dataset_name] = next_label_shift
            next_label_shift += next_label_shift + self.train_generative_ds_dict[dataset_name].class_num

        if self.args.generative_dataset_shared_loader:
            # These two dataloader iters are shared
            self.train_generative_iter_dict = {}
            self.test_generative_iter_dict = {}

    def generate_orthogonal_random_matrix(self):
        logging.info(f"Generating orthogonal_random_matrix, Calculating.............")
        self.VHL_mapping_matrix = orthogo_tensor(
            raw_dim=self.args.model_feature_dim,
            column_dim=self.args.model_feature_dim)
        logging.info(f"Generating orthogonal_random_matrix, validating: det of matrix: "+
                    f"{torch.det(self.VHL_mapping_matrix)}")

    def get_model_bn(self):
        all_bn_params = get_all_bn_params(self.model)
        return all_bn_params

    def set_model_bn(self, all_bn_params):
        # logging.info(f"all_bn_params.keys(): {all_bn_params.keys()}")
        # for name, params in all_bn_params.items():
            # logging.info(f"name:{name}, params.shape: {params.shape}")
        for module_name, module in self.model.named_modules():
            if type(module) is nn.BatchNorm2d:
                # logging.info(f"module_name:{module_name}, params.norm: {module.weight.data.norm()}")
                module.weight.data = all_bn_params[module_name+".weight"] 
                module.bias.data = all_bn_params[module_name+".bias"] 
                module.running_mean = all_bn_params[module_name+".running_mean"] 
                module.running_var = all_bn_params[module_name+".running_var"] 
                module.num_batches_tracked = all_bn_params[module_name+".num_batches_tracked"] 


    def get_model_grads(self):
        named_grads = get_named_data(self.model, mode='GRAD', use_cuda=True)
        # logging.info(f"Getting grads as named_grads: {named_grads}")
        return named_grads

    def set_grad_params(self, named_grads):
        # pass
        self.model.train()
        self.optimizer.zero_grad()
        for name, parameter in self.model.named_parameters():
            parameter.grad.copy_(named_grads[name].data.to(self.device))


    def clear_grad_params(self):
        self.optimizer.zero_grad()

    def update_model_with_grad(self):
        self.model.to(self.device)
        self.optimizer.step()

    def get_optim_state(self):
        return self.optimizer.state


    def clear_optim_buffer(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()


    def lr_schedule(self, progress):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(progress)
        else:
            logging.info("No lr scheduler...........")


    def warmup_lr_schedule(self, iterations):
        if self.lr_scheduler is not None:
            self.lr_scheduler.warmup_step(iterations)

    # Used for single machine training
    # Should be discarded #TODO
    def train(self, train_data, device, args, **kwargs):
        model = self.model

        model.train()

        epoch_loss = []
        for epoch in range(args.max_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # logging.info(images.shape)
                x, labels = x.to(device), labels.to(device)
                self.optimizer.zero_grad()

                if self.args.model_out_feature:
                    output, feat = model(x)
                else:
                    output = model(x)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                logging.info('Training Epoch: {} iter: {} \t Loss: {:.6f}'.format(
                                epoch, batch_idx, loss.item()))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Trainer_ID {}. Train Epo: {} \tLoss: {:.6f}'.format(
                    self.index, epoch, sum(epoch_loss) / len(epoch_loss)))
            self.lr_scheduler.step(epoch=epoch + 1)


    def get_train_batch_data(self, train_local):
        try:
            train_batch_data = self.train_local_iter.next()
            # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_local_iter = iter(train_local)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data


    def summarize(self, model, output, labels,
        tracker, metrics,
        loss,
        epoch, batch_idx,
        mode='train',
        checkpoint_extra_name="centralized",
        things_to_track=[],
        if_update_timer=False,
        train_data=None, train_batch_data=None,
        end_of_epoch=None,
    ):
        # if np.isnan(loss.item()):
        # logging
        if np.isnan(loss):
            logging.info('(WARNING!!!!!!!! Trainer_ID {}. Train epoch: {},\
                iteration: {}, loss is nan!!!! '.format(
                self.index, epoch, batch_idx))
            # loss.data.fill_(100)
            loss = 100
        metric_stat = metrics.evaluate(loss, output, labels)
        tracker.update_metrics(
            metric_stat, 
            metrics_n_samples=labels.size(0)
        )

        if len(things_to_track) > 0:
            if end_of_epoch is not None:
                pass
            else:
                end_of_epoch = (batch_idx == len(train_data) - 1)
            self.track(tracker, self.args.batch_size, model, loss, end_of_epoch,
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track)

        if if_update_timer:
            """
                Now, the client timer need to be updated by each iteration, 
                so as to record the track infomation.
                But only for epoch training, because One-step training will be scheduled by client or server
            """
            tracker.timer.past_iterations(iterations=1)

        if mode == 'train':
            logging.info('Trainer {}. Glob comm round: {}, Train Epo: {}, iter: {} '.format(
                self.index, tracker.timer.global_comm_round_idx, epoch, batch_idx) + metrics.str_fn(metric_stat))
            # logging.info('(Trainer_ID {}. Local Training Epoch: {}, Iter: {} \tLoss: {:.6f} ACC1:{}'.format(
            #     self.index, epoch, batch_idx, sum(batch_loss) / len(batch_loss), metric_stat['Acc1']))
            pass
        elif mode == 'test':
            # logging.info('(Trainer_ID {}. Test epoch: {}, iteration: {} '.format(
            #     self.index, epoch, batch_idx) + metrics.str_fn(metric_stat))
            pass
        else:
            raise NotImplementedError
        return metric_stat
    
    def train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                        tracker=None, metrics=None,
                        local_iterations=None,
                        move_to_gpu=True, make_summary=True,
                        clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                        hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
                        grad_per_sample=False,
                        fim_tr=False, fim_tr_emp=False,
                        parameters_crt_names=[],
                        checkpoint_extra_name="centralized",
                        things_to_track=[],
                        **kwargs):
        model = self.model
        if move_to_gpu:
            model.to(device)
        model.train()
        
        # expect bn not change when training pluggable
        if self.args.freeze_bn:
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
            # freeze BN
            for layer in flatten_submodules(list(model.children())[:-self.args.freeze_layer]):
                if isinstance(layer, torch.nn.BatchNorm2d) and self.args.freeze_bn:
                    # logging.info(f"detech Norm Layer {type(layer).__name__}, Freezing..........")
                    layer.eval()
                    
                    
        batch_loss = []
        if local_iterations is None:
            iterations = len(train_data)
        else:
            iterations = local_iterations

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None

        for batch_idx in range(iterations):
            train_batch_data = self.get_train_batch_data(train_data)
            x, labels = train_batch_data

            real_batch_size = labels.shape[0]
            if self.args.TwoCropTransform:
                x = torch.cat([x[0], x[1]], dim=0)
                labels = torch.cat([labels, labels], dim=0)
            x, labels = x.to(device), labels.to(device)
            if clear_grad_bef_opt:
                self.optimizer.zero_grad()

            if self.args.model_out_feature:
                output, feat = model(x)
            else:
                output = model(x)
            
            loss = self.criterion(output, labels)

            if self.args.fed_align:
                loss_align_Gaussianfeature = self.align_feature_loss(feat, labels, real_batch_size)
                loss += self.args.fed_align_alpha * loss_align_Gaussianfeature
                tracker.update_local_record(
                        'losses_track',
                        server_index=self.server_index,
                        client_index=self.client_index,
                        summary_n_samples=real_batch_size*1,
                        args=self.args,
                        loss_align_Gaussfeat=loss_align_Gaussianfeature.item()
                    )
            if self.args.fedprox:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
                for name, param in model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(device)))**2)
                loss += fed_prox_reg

            loss.backward()
            loss_value = loss.item()
            self.optimizer.step()

            if self.args.scaffold:
                c_model_global = kwargs['c_model_global']
                c_model_local = kwargs['c_model_local']
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.lr
                    print("USING LR_SCHEDULER")
                else:
                    current_lr = self.args.lr
                for name, param in model.named_parameters():
                    # logging.debug(f"c_model_global[name].device : {c_model_global[name].device}, \
                    #     c_model_local[name].device : {c_model_local[name].device}")
                    param.data = param.data - current_lr * \
                        check_device((c_model_global[name] - c_model_local[name]), param.data.device)

            logging.debug(f"epoch: {epoch}, Loss is {loss_value}")

            if make_summary and (tracker is not None) and (metrics is not None):
                self.summarize(model, output, labels,
                        tracker, metrics,
                        loss_value,
                        epoch, batch_idx,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=True if self.args.record_dataframe else False,
                        train_data=train_data, train_batch_data=train_batch_data,
                        end_of_epoch=None,
                    )

    def OOD_train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                        tracker=None, metrics=None,
                        local_iterations=None,
                        move_to_gpu=True, make_summary=True,
                        clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                        hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
                        grad_per_sample=False,
                        fim_tr=False, fim_tr_emp=False,
                        parameters_crt_names=[],
                        checkpoint_extra_name="centralized",
                        things_to_track=[],
                        **kwargs):
        assert self.args.OOD_independ_classifier
        model = self.model
        OOD_dete=self.OOD_dete
        if move_to_gpu:
            model.to(device)
            OOD_dete.to(device)
        model.train()
        
        if self.args.freeze_bn:
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
            # freeze BN
            for layer in flatten_submodules(list(model.children())[:-self.args.freeze_layer]):
                if isinstance(layer, torch.nn.BatchNorm2d) and self.args.freeze_bn:
                    # logging.info(f"detech Norm Layer {type(layer).__name__}, Freezing..........")
                    layer.eval()
        
        OOD_dete.train()
        batch_loss = []
        if local_iterations is None:
            iterations = len(train_data)
        else:
            iterations = local_iterations

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None

        if self.args.OOD_feat_noise:
            if self.args.OOD_noise_option=="other_real_data":
                client_index=kwargs["client_index"]
                noise_data=[]
                for i in kwargs["train_data_local_dict"].keys():
                    if i != client_index:
                        noise_data.append(kwargs["train_data_local_dict"][i].dataset.tensors[0])
                noise_data=torch.concat(noise_data, dim=0)
                # print("noise_data.shape===", noise_data.shape)
                # feature_extractor=torch.nn.Sequential(*list(model.children())[:-self.args.freeze_layer]).to(self.device)
                # if self.args.if_eval:
                #     feature_extractor.eval()

        for batch_idx in range(iterations):
            train_batch_data = self.get_train_batch_data(train_data)
            x, labels = train_batch_data

            real_batch_size = labels.shape[0]
            if self.args.TwoCropTransform:
                x = torch.cat([x[0], x[1]], dim=0)
                labels = torch.cat([labels, labels], dim=0)
            x, labels = x.to(device), labels.to(device)
            if clear_grad_bef_opt:
                self.optimizer.zero_grad()
                self.c_optimizer.zero_grad()

            if self.args.model_out_feature:
                output, feat = model(x)
            else:
                output = model(x)

            OOD=OOD_dete(x)
            if self.args.OOD_feat_noise:
                if self.args.OOD_noise_option=="other_real_data":
                    select_sample=np.random.choice(len(noise_data), real_batch_size, replace=False)

                    noise_x=noise_data[select_sample]
                    noise_data=np.delete(noise_data, select_sample, axis=0)
                    # print("noise_x.shape===", noise_x.shape)
                    X=noise_x.to(self.device)
                    noise_out=OOD_dete(X)# train_data include X and y
                    # print("noisy_out", noise_out.shape)
                elif self.args.OOD_noise_option=="random":
                    if self.args.freeze_layer==1:
                        noisy_feat=torch.randn((len(x),self.args.model_feature_dim)).to(self.device)
                        classifier=list(OOD_dete.children())[-1].to(self.device)
                        if self.args.if_eval:
                            classifier.eval()
                        noise_out=classifier(noisy_feat)
                    else:
                        feat_demo=nn.Sequential(*list(OOD_dete.children())[:-self.args.freeze_layer])(x[:1])
                        noisy_feat=torch.randn((len(x),*feat_demo.shape[1:])).to(self.device)
                        classifier_without_linear=nn.Sequential(*list(OOD_dete.children())[-self.args.freeze_layer:-1]).to(self.device)
                        noise_linear_feat=classifier_without_linear(noisy_feat)
                        noise_out=copy.deepcopy(OOD_dete.linear)(noise_linear_feat.view(noise_linear_feat.size(0), -1))
                        
                noise_label=torch.zeros((len(x),)).to(dtype=torch.long).to(self.device)
                OOD=torch.concat([OOD, noise_out])
                OOD_label=torch.concat([labels[...,1], noise_label])
            
            loss = self.criterion(output, labels.to(dtype=torch.long))
            if self.args.OOD_feat_noise:
                OOD_loss=F.cross_entropy(OOD, OOD_label.to(dtype=torch.long))
                res=torch.argmax(OOD,dim=-1)
                print("OOD_acc:::::::",torch.sum(res==OOD_label)/len(OOD_label))
            else:
                OOD_loss=F.cross_entropy(OOD, labels[..., 1].to(dtype=torch.long))
            
            if self.args.fed_align:
                loss_align_Gaussianfeature = self.align_feature_loss(feat, labels, real_batch_size)
                loss += self.args.fed_align_alpha * loss_align_Gaussianfeature
                tracker.update_local_record(
                        'losses_track',
                        server_index=self.server_index,
                        client_index=self.client_index,
                        summary_n_samples=real_batch_size*1,
                        args=self.args,
                        loss_align_Gaussfeat=loss_align_Gaussianfeature.item()
                    )
            if self.args.fedprox:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
                for name, param in model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(device)))**2)
                loss += fed_prox_reg

            loss.backward()
            OOD_loss.backward()
            loss_value = loss.item()
            loss_value+=OOD_loss.item()

            self.optimizer.step()
            self.c_optimizer.step()

            if self.args.scaffold:
                c_model_global = kwargs['c_model_global']
                c_model_local = kwargs['c_model_local']
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.lr
                    print("USING LR_SCHEDULER")
                else:
                    current_lr = self.args.lr
                for name, param in model.named_parameters():
                    # logging.debug(f"c_model_global[name].device : {c_model_global[name].device}, \
                    #     c_model_local[name].device : {c_model_local[name].device}")
                    param.data = param.data - current_lr * \
                        check_device((c_model_global[name] - c_model_local[name]), param.data.device)

            logging.debug(f"epoch: {epoch}, Loss is {loss_value}")

            if make_summary and (tracker is not None) and (metrics is not None):
                self.summarize(model, output, labels,
                        tracker, metrics,
                        loss_value,
                        epoch, batch_idx,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=True if self.args.record_dataframe else False,
                        train_data=train_data, train_batch_data=train_batch_data,
                        end_of_epoch=None,
                    )

    def FedRod_train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                        tracker=None, metrics=None,
                        local_iterations=None,
                        move_to_gpu=True, make_summary=True,
                        clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                        hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
                        grad_per_sample=False,
                        fim_tr=False, fim_tr_emp=False,
                        parameters_crt_names=[],
                        checkpoint_extra_name="centralized",
                        things_to_track=[],
                        **kwargs):
        def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
            """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
            Args:
            labels: A int tensor of size [batch].
            logits: A float tensor of size [batch, no_of_classes].
            sample_per_class: A int tensor of size [no of classes].
            reduction: string. One of "none", "mean", "sum"
            Returns:
            loss: A float tensor. Balanced Softmax Loss.
            """
            spc = sample_per_class.type_as(logits)
            spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
            logits = logits + spc.log()
            loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
            return loss
        self.model.to(device)
        self.pred.to(device)
        self.model.train()
        self.pred.train()
        batch_loss = []
        if local_iterations is None:
            iterations = len(train_data)
        else:
            iterations = local_iterations

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None

        for batch_idx in range(iterations):
            train_batch_data = self.get_train_batch_data(train_data)
            x, labels = train_batch_data

            real_batch_size = labels.shape[0]
            if self.args.TwoCropTransform:
                x = torch.cat([x[0], x[1]], dim=0)
                labels = torch.cat([labels, labels], dim=0)
            x, labels = x.to(device), labels.to(device)
            
            rep = self.model.base(x)
            out_g = self.model.predictor(rep)
            loss_bsm = balanced_softmax_loss(labels, out_g, self.sample_per_class)
            self.optimizer.zero_grad()
            loss_bsm.backward()
            self.optimizer.step()

            out_p = self.pred(rep.detach())
            loss = self.criterion(out_g.detach() + out_p, labels)
            self.opt_pred.zero_grad()
            loss.backward()
            self.opt_pred.step()
            loss_value = loss.item()

            logging.debug(f"epoch: {epoch}, Loss is {loss_value}")

            if make_summary and (tracker is not None) and (metrics is not None):
                self.summarize(self.model, out_g, labels,
                        tracker, metrics,
                        loss_value,
                        epoch, batch_idx,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=True if self.args.record_dataframe else False,
                        train_data=train_data, train_batch_data=train_batch_data,
                        end_of_epoch=None,
                    )

    def FedRep_train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                            tracker=None, metrics=None,
                            local_iterations=None,
                            move_to_gpu=True, make_summary=True,
                            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                            hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
                            grad_per_sample=False,
                            fim_tr=False, fim_tr_emp=False,
                            parameters_crt_names=[],
                            checkpoint_extra_name="centralized",
                            things_to_track=[],
                            **kwargs):
        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.predictor.parameters():
            param.requires_grad = True

        for step in range(self.args.p_local_step):
            for i, (x, y) in enumerate(train_data):
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)
                self.poptimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.poptimizer.step()

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.predictor.parameters():
            param.requires_grad = False

        for step in range(self.args.global_epochs_per_round):
            for i, (x, y) in enumerate(train_data):
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                loss_value=loss.item()
                self.optimizer.step()

                logging.debug(f"epoch: {epoch}, Loss is {loss_value}")

                if make_summary and (tracker is not None) and (metrics is not None):
                    self.summarize(self.model, output, y,
                            tracker, metrics,
                            loss_value,
                            epoch, i,
                            mode='train',
                            checkpoint_extra_name=checkpoint_extra_name,
                            things_to_track=things_to_track,
                            if_update_timer=True if self.args.record_dataframe else False,
                            train_data=train_data, train_batch_data=(x, y),
                            end_of_epoch=None,
                        )
    
    def FedTHE_train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                        tracker=None, metrics=None,
                        local_iterations=None,
                        move_to_gpu=True, make_summary=True,
                        clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                        hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
                        grad_per_sample=False,
                        fim_tr=False, fim_tr_emp=False,
                        parameters_crt_names=[],
                        checkpoint_extra_name="centralized",
                        things_to_track=[],
                        **kwargs):
        def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
            """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
            Args:
            labels: A int tensor of size [batch].
            logits: A float tensor of size [batch, no_of_classes].
            sample_per_class: A int tensor of size [no of classes].
            reduction: string. One of "none", "mean", "sum"
            Returns:
            loss: A float tensor. Balanced Softmax Loss.
            """
            spc = sample_per_class.type_as(logits)
            spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
            logits = logits + spc.log()
            loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
            return loss
        self.model.to(device)
        self.pred.to(device)
        self.model.train()
        self.pred.train()
        batch_loss = []
        if local_iterations is None:
            iterations = len(train_data)
        else:
            iterations = local_iterations

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None

        for batch_idx in range(iterations):
            train_batch_data = self.get_train_batch_data(train_data)
            x, labels = train_batch_data

            real_batch_size = labels.shape[0]
            if self.args.TwoCropTransform:
                x = torch.cat([x[0], x[1]], dim=0)
                labels = torch.cat([labels, labels], dim=0)
            x, labels = x.to(device), labels.to(device)
            
            rep = self.model.base(x)
            out_g = self.model.predictor(rep)
            loss_bsm = balanced_softmax_loss(labels, out_g, self.sample_per_class)
            self.optimizer.zero_grad()
            loss_bsm.backward()
            self.optimizer.step()

            out_p = self.pred(rep.detach())
            loss = self.criterion(out_p, labels)
            self.opt_pred.zero_grad()
            loss.backward()
            self.opt_pred.step()
            loss_value = loss.item()

            self.per_class_rep = {i: [] for i in range(args.num_classes)}
            for i, label in enumerate(labels):
                self.per_class_rep[label.item()].append(rep[i, :].unsqueeze(0))
            
            self.local_rep = []
            for (k, v) in self.per_class_rep.items():
                if len(v) != 0:
                    self.local_rep.append(torch.cat(v).cuda(self.device))
            self.local_rep = torch.cat(self.local_rep).mean(dim=0).cuda(self.device)
            logging.debug(f"epoch: {epoch}, Loss is {loss_value}")

            if make_summary and (tracker is not None) and (metrics is not None):
                self.summarize(self.model, out_g, labels,
                        tracker, metrics,
                        loss_value,
                        epoch, batch_idx,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=True if self.args.record_dataframe else False,
                        train_data=train_data, train_batch_data=train_batch_data,
                        end_of_epoch=None,
                    )
    
    def FedSAM_train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                        tracker=None, metrics=None,
                        local_iterations=None,
                        move_to_gpu=True, make_summary=True,
                        clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                        hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
                        grad_per_sample=False,
                        fim_tr=False, fim_tr_emp=False,
                        parameters_crt_names=[],
                        checkpoint_extra_name="centralized",
                        things_to_track=[],
                        **kwargs):
        model = self.model
        if move_to_gpu:
            model.to(device)
        model.train()
                    
        batch_loss = []
        if local_iterations is None:
            iterations = len(train_data)
        else:
            iterations = local_iterations

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None

        for batch_idx in range(iterations):
            train_batch_data = self.get_train_batch_data(train_data)
            x, labels = train_batch_data

            real_batch_size = labels.shape[0]
            if self.args.TwoCropTransform:
                x = torch.cat([x[0], x[1]], dim=0)
                labels = torch.cat([labels, labels], dim=0)
            x, labels = x.to(device), labels.to(device)

            if self.args.model_out_feature:
                output, feat = model(x)
            else:
                output = model(x)
            
            loss = self.criterion(output, labels)

            loss.backward()
            self.optimizer.ascent_step()
            
            self.criterion(model(x), labels).backward()
            self.optimizer.descent_step()
            
            loss_value = loss.item()

            logging.debug(f"epoch: {epoch}, Loss is {loss_value}")

            if make_summary and (tracker is not None) and (metrics is not None):
                self.summarize(model, output, labels,
                        tracker, metrics,
                        loss_value,
                        epoch, batch_idx,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=True if self.args.record_dataframe else False,
                        train_data=train_data, train_batch_data=train_batch_data,
                        end_of_epoch=None,
                    )
    
    def FedTHE_test(self, test_data, device=None, args=None):
        self.model.eval()
        self.model.to(device)
        self.pred.to(device)
        self.is_in_personalized_training = True
        # dont requires gradients.
        self.model.requires_grad_(False)
        self.pred.requires_grad_(False)
        self.model.eval()
        
        iterations = len(test_data)
        
        # if enabled, then the test history is reused between test sets.
        self.test_history = None
        
        all_correct = 0
        sample_num = 0
        
        for test_batch_data in test_data:
            x, labels = test_batch_data
            x, labels = x.to(device), labels.to(device)
            with torch.no_grad():
                # update test history by exponential moving average
                test_rep = self.model.base(x).detach()
                test_history = None
                for i in range(test_rep.shape[0]):
                    if test_history is None and self.test_history is None:
                        test_history = [test_rep[0, :]]
                    elif test_history is None and self.test_history is not None:
                        test_history = [self.test_history[-1, :]]
                    else:
                        test_history.append(self.args.FedTHE_alpha * test_rep[i, :] + (1 - self.args.FedTHE_alpha) * test_history[-1])
                self.test_history = torch.stack(test_history)
                temperature = torch.hstack((torch.ones((test_rep.shape[0], 1)).cuda(self.device), torch.ones((test_rep.shape[0], 1)).cuda(self.device)))
            
            self.agg_weight = torch.nn.parameter.Parameter(torch.tensor(temperature).cuda(self.device), requires_grad=True)
            self.agg_optim = torch.optim.Adam([self.agg_weight], lr=self.args.lr)

            self._calculate_samplewise_weight(self.model, test_batch_data, 20)
            
            if self.args.FedTHE_finetune:
                # test-timely tune the whole net. (w FedTHE+ / o FedTHE)
                g_pred, p_pred = self._test_time_tune(self.model, test_batch_data, num_steps=10)

            # do inference for current batch
            with torch.no_grad():
                if self.args.FedTHE_finetune:
                    correct,_ = self._multi_head_inference(test_batch_data, self.model, g_pred, p_pred)
                else:
                    correct,_ = self._multi_head_inference(test_batch_data, self.model)
            
            all_correct += correct
            sample_num += len(x)
        
        self.is_in_personalized_training = False
        self.agg_weight.data.fill_(1 / 2)
        
        return all_correct, sample_num
    
    def _calculate_samplewise_weight(self, model, test_batch_data, num_epochs):
        # function that optimize the ensemble weights.
        x, labels = test_batch_data
        x, labels = x.to(self.device), labels.to(self.device)
        self.model.to(self.device)
        model.to(self.device)
        self.pred.to(self.device)
        test_rep = model.base(x)
        g_out = model.predictor(test_rep)
        p_out = self.pred(test_rep)
        grad_norm, loss_batch = [], []
        for _ in range(num_epochs):
            # normalize the aggregation weight by softmax
            agg_softmax = torch.nn.functional.softmax(self.agg_weight)
            agg_output = agg_softmax[:, 0].unsqueeze(1) * g_out.detach() \
                         + agg_softmax[:, 1].unsqueeze(1) * p_out.detach()
            # formulate test representation.
            test_rep = self.args.FedTHE_beta * test_rep + (1 - self.args.FedTHE_beta) * self.test_history
            p_feat_al = torch.norm((test_rep - self.local_rep.to(self.device)), dim=1)
            g_feat_al = torch.norm((test_rep - self.global_rep.to(self.device)), dim=1)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos(F.softmax(g_out).detach(), F.softmax(p_out).detach())
            # loss function based on prediction similarity, entropy minimization and feature alignment.
            loss = (-sim * (agg_output.softmax(1) * agg_output.log_softmax(1)).sum(1) + \
                    (1 - sim) * (agg_softmax[:, 0] * g_feat_al.detach() + agg_softmax[:, 1] * p_feat_al.detach())).mean(0)
            self.agg_optim.zero_grad()
            loss.backward()

            if torch.norm(self.agg_weight.grad) < 1e-5:
                break
            grad_norm.append(torch.norm(self.agg_weight.grad).item())
            loss_batch.append(loss.item())
            self.agg_optim.step()
    
    def _multi_head_inference(self, test_batch_data, model, tracker=None, g_pred=None, p_pred=None):
        # inference procedure for multi-head nets.
        x, labels = test_batch_data
        x, labels = x.to(self.device), labels.to(self.device)
        self.model.to(self.device)
        model.to(self.device)
        self.pred.to(self.device)
        agg_softmax = torch.nn.functional.softmax(self.agg_weight)
        if g_pred is not None and p_pred is not None:
            agg_output = agg_softmax[:, 0].unsqueeze(1) * g_pred \
                         + agg_softmax[:, 1].unsqueeze(1) * p_pred
        else:
            # do the forward pass and get the output.
            rep = model.base(x)
            g_out = model.predictor(rep)
            p_out = self.pred(rep)
            agg_output = agg_softmax[:, 0].unsqueeze(1) * g_out \
                         + agg_softmax[:, 1].unsqueeze(1) * p_out
        # evaluate the output and get the loss, performance.
        loss = self.criterion(agg_output, labels)
        correct = (torch.sum(torch.argmax(agg_output, dim=1) == labels)).item()
        # performance = self.metrics.evaluate(loss, agg_output, labels)

        # update tracker.
        # if tracker is not None:
        #     tracker.update_metrics(
        #         [loss.item()] + performance, n_samples=labels.size(0)
        #     )

        return correct, loss
    
    def _test_time_tune(self, model, test_batch_data, num_steps=3):\
        # turn on model grads.
        def get_normalization(data_name):
            if data_name == "cifar10":
                normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.247, 1/0.243, 1/0.261]),
                                                transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465], std=[1., 1., 1.])])
                return normalize, unnormalize
            elif data_name == "cifar100":
                normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.2675, 1/0.2565, 1/0.2761]),
                                                transforms.Normalize(mean=[-0.5071, -0.4867, -0.4408], std=[1., 1., 1.])])
                return normalize, unnormalize
            elif "imagenet" in data_name:
                normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
                                                transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.])])
                # normalize = transforms.Normalize((0.4810, 0.4574, 0.4078), (0.2146, 0.2104, 0.2138))
                # unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.2146, 1/0.2104, 1/0.2138]),
                #                                   transforms.Normalize(mean=[-0.4810, -0.4574, -0.4078], std=[1., 1., 1.])])
                return normalize, unnormalize
        def marginal_entropy(outputs):
            logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
            avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
            min_real = torch.finfo(avg_logits.dtype).min
            avg_logits = torch.clamp(avg_logits, min=min_real)
            return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
        from utils.aug_data import aug
        x, labels = test_batch_data
        x, labels = x.to(self.device), labels.to(self.device)
        self.model.to(self.device)
        model.to(self.device)
        self.pred.to(self.device)
        model.requires_grad_(True)
        self.pred.requires_grad_(True)
        # set optimizer.
        fe_optim = torch.optim.SGD(model.parameters(), lr=0.0005)
        fe_optim.add_param_group({"params": self.pred.parameters()})
        g_pred, p_pred = [], []
        # do the unnormalize to ensure consistency.
        normalize, unnormalize = get_normalization(self.args.dataset)
        convert_img = transforms.Compose([unnormalize, transforms.ToPILImage()])
        agg_softmax = torch.nn.functional.softmax(self.agg_weight).detach()
        model_param = copy.deepcopy(model.state_dict())
        p_head_param = copy.deepcopy(self.pred.state_dict())
        for i in range(x.shape[0]):
            image = convert_img(x[i])
            for _ in range(num_steps):
                # generate a batch of augmentations and minimize prediction entropy.
                inputs = [aug(image, normalize) for _ in range(16)]
                inputs = torch.stack(inputs).cuda(self.device)
                fe_optim.zero_grad()
                rep = model.base(inputs)
                g_out = model.predictor(rep)
                p_out = self.pred(rep)
                agg_output = agg_softmax[i, 0] * g_out + agg_softmax[i, 1] * p_out
                loss, _ = marginal_entropy(agg_output)
                loss.backward()
                fe_optim.step()
            with torch.no_grad():
                rep = model.base(x[i].unsqueeze(0).cuda(self.device))
                g_out = model.predictor(rep)
                p_out = self.pred(rep)
                g_pred.append(g_out)
                p_pred.append(p_out)
            model.load_state_dict(model_param)
            self.pred.load_state_dict(p_head_param)
        # turn off grads.
        model.requires_grad_(False)
        self.pred.requires_grad_(False)
        return torch.cat(g_pred), torch.cat(p_pred)
    
    def FedRod_test(self, test_data, device=None, args=None, epoch=None,
            tracker=None, metrics=None,
            move_to_gpu=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs):
        from sklearn.preprocessing import label_binarize
        self.model.to(device)
        self.pred.to(device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in test_data:
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)
                rep = self.model.base(x)
                out_g = self.model.predictor(rep)
                out_p = self.pred(rep.detach())
                output = out_g.detach() + out_p

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.args.num_classes
                if self.args.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.args.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return test_acc, test_num

    def generate_noise_data(self, noise_label_style="extra", y_train=None):
        # means = torch.zeros((self.args.batch_size, self.args.fedaux_noise_size))
        # noise = torch.normal(mean=means, std=1.0)

        noise_label_shift = 0

        if noise_label_style == "extra":
            noise_label_shift = self.args.num_classes
            chunk_num = self.args.VHL_num
            chunk_size = self.args.batch_size // chunk_num
            # chunks = np.ones(chunk_num)* chunk_size
            chunks = [chunk_size] * chunk_num
            for i in range(self.args.batch_size - chunk_num * chunk_size):
                chunks[i] += 1
        elif noise_label_style == "patch":
            noise_label_shift = 0
            if self.args.VHL_data == "dataset" and self.args.VHL_label_from == "dataset":
                chunk_num = self.args.num_classes
                bs = y_train.shape[0]
                batch_cls_counts = record_batch_data_stats(y_train, bs=bs, num_classes=self.args.num_classes)
                inverse_label_weights = [ bs / (num_label+1)  for label, num_label in batch_cls_counts.items()]
                sum_weights = sum(inverse_label_weights)

                noise_label_weights = [ label_weight / sum_weights for i, label_weight in enumerate(inverse_label_weights)]
                chunks = [int(noise_label_weight *self.args.batch_size)  for noise_label_weight in noise_label_weights]
            else:
                pass
        else:
            raise NotImplementedError

        if self.args.VHL_data == "dataset" and self.args.VHL_label_from == "dataset":

            noise_data_list = []
            noise_data_labels = []
            # In order to implement traverse the extra datasets, automatically generate iterator.
            for dataset_name, train_generative_dl in self.train_generative_dl_dict.items():
                # train_batch_data = get_train_batch_data(self.train_generative_iter_dict, dataset_name,
                #     train_generative_dl, batch_size=self.args.batch_size / len(self.train_generative_dl_dict))
                train_batch_data = get_train_batch_data(self.train_generative_iter_dict, dataset_name,
                    train_generative_dl,
                    batch_size=self.args.VHL_dataset_batch_size / len(self.train_generative_dl_dict))
                logging.debug(f"id(self.train_generative_iter_dict) : {id(self.train_generative_iter_dict)}")
                data, label = train_batch_data
                # logging.debug(f"data.shape: {data.shape}")
                noise_data_list.append(data)
                label_shift = self.noise_dataset_label_shift[dataset_name] + noise_label_shift
                noise_data_labels.append(label + label_shift)
            noise_data = torch.cat(noise_data_list).to(self.device)
            labels = torch.cat(noise_data_labels).to(self.device)
        else:
            raise NotImplementedError

        if self.args.VHL_data_re_norm:
            noise_data = noise_data / 0.25 * 0.5

        return noise_data, labels

    def VHL_train_generator(self):
        for i in range(self.args.VHL_generator_num):
            generator = self.generator_dict[i]
            self.train_generator_diversity(generator, 50)
            # self.train_generator_diversity(generator, 5)

    def train_generator_diversity(self, generator, max_iters=100, min_loss=0.0):
        generator.train()
        generator.to(self.device)
        for i in range(max_iters):
            generator_optimizer = torch.optim.SGD(generator.parameters(),
                lr=0.01, weight_decay=0.0001, momentum=0.9)
            means = torch.zeros((64, self.args.fedaux_noise_size))
            z = torch.normal(mean=means, std=1.0).to(self.device)
            data = generator(z)
            loss_diverse = cov_non_diag_norm(data)
            generator_optimizer.zero_grad()
            loss_diverse.backward()
            generator_optimizer.step()
            logging.info(f"Iteration: {i}, loss_diverse: {loss_diverse.item()}")
            if loss_diverse.item() < min_loss:
                logging.info(f"Iteration: {i}, loss_diverse: {loss_diverse.item()} smaller than min_loss: {min_loss}, break")
                break
        generator.cpu()

    def VHL_get_diverse_distribution(self):
        n_dim = self.generator_dict[0].num_layers
        normed_n_mean = train_distribution_diversity(
            n_distribution=self.args.VHL_num, n_dim=n_dim, max_iters=500)
        self.style_GAN_latent_noise_mean = normed_n_mean.detach()
        self.style_GAN_latent_noise_std = [0.1 / n_dim]*n_dim

        global_zeros = torch.ones((self.args.VHL_num, self.args.style_gan_style_dim)) * 0.0
        global_mean_vector = torch.normal(mean=global_zeros, std=self.args.style_gan_sample_z_mean)
        self.style_GAN_sample_z_mean = global_mean_vector
        self.style_GAN_sample_z_std = self.args.style_gan_sample_z_std

    def VHL_train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                        tracker=None, metrics=None,
                        local_iterations=None,
                        move_to_gpu=True, make_summary=True,
                        clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                        hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
                        grad_per_sample=False,
                        fim_tr=False, fim_tr_emp=False,
                        parameters_crt_names=[],
                        checkpoint_extra_name="centralized",
                        things_to_track=[],
                        **kwargs):
        model = self.model

        time_table = {}

        if move_to_gpu:
            # self.generator.to(device)
            model.to(device)

        # self.generator.train()
        model.train()
        # self.generator.eval()

        batch_loss = []
        if local_iterations is None:
            iterations = len(train_data)
        else:
            iterations = local_iterations

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]

        # for batch_idx, (x, labels) in enumerate(train_data):
        for batch_idx in range(iterations):
            train_batch_data = self.get_train_batch_data(train_data)
            x, labels = train_batch_data
            # if batch_idx > 5:
            #     break
            x, labels = x.to(device), labels.to(device)
            if clear_grad_bef_opt:
                self.optimizer.zero_grad()

            time_now = time.time()
            real_batch_size = labels.shape[0]

            aux_data, sampled_label = self.generate_noise_data(
                noise_label_style=self.args.VHL_label_style, y_train=labels)
            sampled_label = sampled_label.to(device)
            if x.shape[1] == 1:
                assert self.args.dataset in ["mnist", "femnist", "fmnist", "femnist-digit"]
                x = x.repeat(1, 3, 1, 1)
            # logging.info(f"x.shape: {x.shape}, aux_data.shape: {aux_data.shape} " )
            x_cat = torch.cat((x, aux_data), dim=0)

            # y_cat = torch.cat((labels, labels), dim=0)
            if self.args.model_out_feature:
                output, feat = model(x_cat)
            else:
                output = model(x_cat)

            loss_origin = F.cross_entropy(output[0:real_batch_size], labels)
            loss_aux = F.cross_entropy(output[real_batch_size:], sampled_label)
            # loss = (1 - alpha) * loss_origin + alpha * loss_aux
            loss = loss_origin + self.args.VHL_alpha * loss_aux
            loss_origin_value = loss_origin.item()

            align_domain_loss_value = 0.0
            align_cls_loss_value = 0.0
            noise_cls_loss_value = 0.0
            if self.args.VHL_feat_align and epoch < self.args.VHL_align_local_epoch:
                loss_feat_align, align_domain_loss_value, align_cls_loss_value, noise_cls_loss_value = self.proxy_align_loss(
                    feat, torch.cat([labels, sampled_label], dim=0), real_batch_size)
                loss += loss_feat_align

            if self.args.fedprox:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
                for name, param in model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(device)))**2)
                loss += fed_prox_reg

            # loss = F.cross_entropy(output, y_cat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.args.scaffold:
                c_model_global = kwargs['c_model_global']
                c_model_local = kwargs['c_model_local']
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.lr
                else:
                    current_lr = self.args.lr
                for name, param in model.named_parameters():
                    # logging.debug(f"c_model_global[name].device : {c_model_global[name].device}, \
                    #     c_model_local[name].device : {c_model_local[name].device}")
                    param.data = param.data - current_lr * \
                        check_device((c_model_global[name] - c_model_local[name]), param.data.device)

            logging.debug(f"epoch: {epoch}, Loss is {loss.item()}")

            metric_stat = metrics.evaluate(loss_aux, output[real_batch_size:], sampled_label)
            tracker.update_local_record(
                    'generator_track',
                    server_index=self.server_index,
                    client_index=self.client_index,
                    summary_n_samples=real_batch_size*1,
                    args=self.args,
                    Loss=metric_stat["Loss"],
                    Acc1=metric_stat["Acc1"],
                    align_domain_loss_value=align_domain_loss_value,
                    align_cls_loss_value=align_cls_loss_value,
                    noise_cls_loss_value=noise_cls_loss_value,
                )


            if make_summary and (tracker is not None) and (metrics is not None):
                self.summarize(model, output[0:real_batch_size], labels,
                        tracker, metrics,
                        loss_origin_value,
                        epoch, batch_idx,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=True if self.args.record_dataframe else False,
                        train_data=train_data, train_batch_data=train_batch_data,
                        end_of_epoch=None,
                    )
        return loss, output, labels, x_cat

    # train a single step in machine learning
    def train_one_step(self, train_batch_data, device=None, args=None,
            epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None,
            move_to_gpu=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
            grad_per_sample=False,
            fim_tr=False, fim_tr_emp=False,
            parameters_crt_names=[],
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs
        ):

        model = self.model

        if move_to_gpu:
            model.to(device)

        model.train()

        x, labels = train_batch_data

        if self.args.TwoCropTransform:
            x = torch.cat([x[0], x[1]], dim=0)
            labels = torch.cat([labels, labels], dim=0)

        x, labels = x.to(device), labels.to(device)
        real_batch_size = labels.shape[0]

        if clear_grad_bef_opt:
            self.optimizer.zero_grad()

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None

        if self.args.VHL:
            aux_data, sampled_label = self.generate_noise_data(
                noise_label_style=self.args.VHL_label_style, y_train=labels)
            # sampled_label = torch.full((self.args.batch_size), self.args.num_classes).long()
            # sampled_label = (torch.ones(self.args.batch_size)*self.args.num_classes).long().to(device)
            sampled_label = sampled_label.to(device)
            # self.generator.eval()

            if x.shape[1] == 1:
                assert self.args.dataset in ["mnist", "femnist", "fmnist", "femnist-digit"]
                x = x.repeat(1, 3, 1, 1)

            x_cat = torch.cat((x, aux_data), dim=0)

            if self.args.model_out_feature:
                output, feat = model(x_cat)
            else:
                output = model(x_cat)
            loss_origin = F.cross_entropy(output[0:real_batch_size], labels)
            loss_aux = F.cross_entropy(output[real_batch_size:], sampled_label)
            # loss = (1 - alpha) * loss_origin + alpha * loss_aux
            loss = loss_origin + self.args.VHL_alpha * loss_aux
            loss_origin_value = loss_origin.item()

            align_domain_loss_value = 0.0
            align_cls_loss_value = 0.0
            noise_cls_loss_value = 0.0
            if self.args.VHL_feat_align and epoch < self.args.VHL_align_local_epoch:
                loss_feat_align, align_domain_loss_value, align_cls_loss_value, noise_cls_loss_value = self.proxy_align_loss(
                    feat, torch.cat([labels, sampled_label], dim=0), real_batch_size)
                loss += loss_feat_align
        else:
            output = model(x)
        loss = self.criterion(output, labels)

        if self.args.fed_align:
            loss_align_Gaussianfeature = self.align_feature_loss(feat, labels, real_batch_size)
            loss += self.args.fed_align_alpha * loss_align_Gaussianfeature
            tracker.update_local_record(
                    'losses_track',
                    server_index=self.server_index,
                    client_index=self.client_index,
                    summary_n_samples=real_batch_size*1,
                    args=self.args,
                    loss_align_Gaussfeat=loss_align_Gaussianfeature.item()
                )

        loss.backward()
        loss_value = loss.item()

        self.optimizer.step()

        if make_summary and (tracker is not None) and (metrics is not None):
            # logging.info(f"")
            self.summarize(model, output, labels,
                    tracker, metrics,
                    loss_value,
                    epoch, iteration,
                    mode='train',
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track,
                    if_update_timer=False,
                    train_data=None, train_batch_data=train_batch_data,
                    end_of_epoch=end_of_epoch,
                )

        return loss, output, labels

    def infer_bw_one_step(self, train_batch_data, device=None, args=None,
            epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None,
            move_to_gpu=True, model_train=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
            grad_per_sample=False,
            fim_tr=False, fim_tr_emp=False,
            parameters_crt_names=[],
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs
        ):
        """
            inference and BP without optimization
        """
        model = self.model

        if move_to_gpu:
            model.to(device)

        if model_train:
            model.train()
        else:
            model.eval()

        time_table = {}
        time_now = time.time()
        x, labels = train_batch_data
        x, labels = x.to(device), labels.to(device)

        if clear_grad_bef_opt:
            self.optimizer.zero_grad()

        if self.args.model_out_feature:
            output, feat = model(x)
        else:
            output = model(x)
        loss = self.criterion(output, labels)
        loss_value = loss.item()
        time_table["FP"] = time.time() - time_now
        time_now = time.time()
        logging.debug(f" Whole model time FP: {time.time() - time_now}")

        loss.backward()

        if make_summary and (tracker is not None) and (metrics is not None):
            self.summarize(model, output, labels,
                    tracker, metrics,
                    loss_value,
                    epoch, iteration,
                    mode='train',
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track,
                    if_update_timer=False,
                    train_data=None, train_batch_data=train_batch_data,
                    end_of_epoch=end_of_epoch,
                )

        return loss, output, labels

    # test for one model
    def test(self, test_data, device=None, args=None, epoch=None,
            tracker=None, metrics=None,
            move_to_gpu=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs):

        model = self.model
        Acc_accm = 0.0

        model.eval()
        if move_to_gpu:
            model.to(device)
        real_datanum=0
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_data):
                x = x.to(device)
                labels = labels.to(device)
                real_batch_size = labels.shape[0]
                real_datanum+=real_batch_size
                if self.args.model_input_channels == 3 and x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if self.args.model_out_feature:
                    output, feat = model(x)
                else:
                    output = model(x)

                loss = self.criterion(output, labels)
                if self.args.VHL and self.args.VHL_shift_test:
                    if self.args.VHL_label_style == "patch":
                        noise_label_shift = 0
                    elif self.args.VHL_label_style == "extra":
                        noise_label_shift = self.args.num_classes
                    else:
                        raise NotImplementedError
                        
                    metric_stat = metrics.evaluate(loss, output, labels, pred_shift=noise_label_shift)
                    tracker.update_local_record(
                            'generator_track',
                            server_index=self.server_index,
                            client_index=self.client_index,
                            summary_n_samples=labels.shape[0]*1,
                            args=self.args,
                            PredShift_Loss=metric_stat["Loss"],
                            PredShift_Acc1=metric_stat["Acc1"],
                        )

                if make_summary and (tracker is not None) and (metrics is not None):
                    metric_stat = self.summarize(model, output, labels,
                            tracker, metrics,
                            loss.item(),
                            epoch, batch_idx,
                            mode='test',
                            checkpoint_extra_name=checkpoint_extra_name,
                            things_to_track=things_to_track,
                            if_update_timer=False,
                            train_data=test_data, train_batch_data=None,
                            end_of_epoch=False,
                        )
                    logging.debug(f"metric_stat[Acc1] is {metric_stat['Acc1']} ")
                    Acc_accm += metric_stat["Acc1"]*real_batch_size
            logging.debug(f"Total is {Acc_accm}, Averaged is {Acc_accm/real_datanum}")
        # if self.args.HPFL:
        return Acc_accm / real_datanum

    def OOD_test(self, test_data, device=None, args=None, epoch=None,
            tracker=None, metrics=None,
            move_to_gpu=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs):
        model = self.model
        if self.args.OOD_independ_classifier:
            OOD_dete=self.OOD_dete
        Acc_accm = 0.0

        model.eval()
        if self.args.OOD_independ_classifier:
            OOD_dete.eval()
        if move_to_gpu:
            model.to(device)
            if self.args.OOD_independ_classifier:
                OOD_dete.to(device)
        real_datanum=0
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_data):
                x = x.to(device)
                labels = labels.to(device)
                if self.args.model_input_channels == 3 and x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if self.args.model_out_feature:
                    output, feat = model(x)
                else:
                    output = model(x)
                if self.args.OOD_independ_classifier:
                    OOD=OOD_dete(x)
                if self.args.OOD_independ_classifier:
                    loss = self.criterion(output, labels.to(dtype=torch.long))
                    loss+=F.cross_entropy(OOD, labels[..., 1].to(dtype=torch.long))
                else:
                    loss = self.criterion(output, labels)
                if self.args.VHL and self.args.VHL_shift_test:
                    if self.args.VHL_label_style == "patch":
                        noise_label_shift = 0
                    elif self.args.VHL_label_style == "extra":
                        noise_label_shift = self.args.num_classes
                    else:
                        raise NotImplementedError
                        
                    metric_stat = metrics.evaluate(loss, output, labels, pred_shift=noise_label_shift)
                    tracker.update_local_record(
                            'generator_track',
                            server_index=self.server_index,
                            client_index=self.client_index,
                            summary_n_samples=labels.shape[0]*1,
                            args=self.args,
                            PredShift_Loss=metric_stat["Loss"],
                            PredShift_Acc1=metric_stat["Acc1"],
                        )
                real_batch_size = labels[labels[...,-1]==1].shape[0]
                real_datanum+=real_batch_size
                if make_summary and (tracker is not None) and (metrics is not None):
                    metric_stat = self.summarize(model, output, labels,
                            tracker, metrics,
                            loss.item(),
                            epoch, batch_idx,
                            mode='test',
                            checkpoint_extra_name=checkpoint_extra_name,
                            things_to_track=things_to_track,
                            if_update_timer=False,
                            train_data=test_data, train_batch_data=None,
                            end_of_epoch=False,
                        )
                    logging.debug(f"metric_stat[Acc1] is {metric_stat['Acc1']} ")
                    Acc_accm += metric_stat["Acc1"]*real_batch_size
            logging.debug(f"Total is {Acc_accm}, Averaged is {Acc_accm/real_datanum}")
            
        if self.args.HPFL:
            return Acc_accm / real_datanum

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None,
                        epoch=None, iteration=None, tracker=None, metrics=None):
        pass

    def feature_reduce(self, time_stamp=0, reduce_method="tSNE", extra_name="cent", postfix="",
        batch_data=None, data_loader=None, num_points=1000, save_checkpoint=True, save_image=False):
        # time_stamp represents epoch or round.
        data_tsne, labels = None, None
        if time_stamp in self.args.tSNE_track_epoch_list:
            if save_checkpoint:
                save_checkpoint_without_check(
                    self.args, self.save_checkpoints_config,
                    extra_name=extra_name,
                    epoch=time_stamp,
                    model_state_dict=self.get_model_params(),
                    optimizer_state_dict=None,
                    postfix=postfix,
                )
            if save_image:
                data_tsne, labels = self.dim_reducer.unsupervised_reduce(reduce_method=reduce_method, 
                    model=self.model, batch_data=batch_data, data_loader=data_loader, num_points=num_points)
                logging.info(f"data_tsne.shape: {data_tsne.shape}")
                if postfix is not None:
                    postfix_str = "-" + postfix
                else:
                    postfix_str = ""
                image_path = self.args.checkpoint_root_path + \
                    extra_name + setup_checkpoint_file_name_prefix(self.args) + \
                    "-epoch-"+str(time_stamp) + postfix_str +'.jpg'
                # save_image(tensor=x_cat, fp=image_path, nrow=8)
                # logging.info(f"data_tsne.deivce: {data_tsne.deivce}, labels.device: {labels.device} ")
                # logging.info(f"labels.device: {labels.device} ")

                plt.figure(figsize=(6, 4))
                plt.subplot(1, 2, 1)
                plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, alpha=0.6, 
                            cmap=plt.cm.get_cmap('rainbow', 10))
                plt.title("t-SNE")
                plt.savefig(image_path)
        return data_tsne, labels






