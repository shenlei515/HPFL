import argparse
import logging
import os
import random
import socket
import sys
import yaml
import copy

import traceback
# from mpi4py import MPI

import numpy as np
import psutil
import setproctitle
import torch
import wandb
# add the FedML root directory to the python path

from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

# from utils.timer import Timer
from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.wandb_util import wandb_log

from data_preprocessing.build import load_data
from data_preprocessing.random_label import switch_random_labels
from data_preprocessing.loader import Data_Loader

from model.build import create_model
from model.load_submodel import load_submodel

from trainers.build import create_trainer

from trainers.freezer_custom import Freezer_Custom

from utils.logger import (
    logging_config
)

from configs import get_cfg, build_config

from utils.data_utils import (
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations
)

from utils.model_utils import (
    set_freeze_by_names,
    get_actual_layer_names,
    freeze_by_names,
    unfreeze_by_names
)
from utils.checkpoint import (
    setup_checkpoint_config, setup_save_checkpoint_path, save_checkpoint,
    setup_checkpoint_file_name_prefix
)


from utils.checkpoint import setup_checkpoint_config, setup_save_checkpoint_path, save_checkpoint

from timers.server_timer import ServerTimer

from utils.auxiliary import check_and_test


from loss_fn.cov_loss import (
    cov_non_diag_norm, cov_norm
)
from loss_fn.losses import LabelSmoothingCrossEntropy, proxy_align_loss




# def centralized_reprogram(cfg, device):




def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--config_file", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str,
                        help="specify add which type of config")
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # 1 is for optimizer intialization
    process_id = 0
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    #### set up cfg ####
    # default cfg
    cfg = get_cfg()

    # add registered cfg
    # some arguments that are needed by build_config come from args.
    cfg.setup(args)
    build_config(cfg, args.config_name)

    # Build config once again
    cfg.setup(args)
    cfg.mode = 'centralized'

    cfg.rank = 0
    cfg.algorithm = "centralized"
    cfg.role = "server"
    cfg.server_index = 0
    cfg.client_index = 0

    seed = cfg.seed

    # show ultimate config
    logging.info(dict(cfg))

    # customize the process name
    str_process_name = cfg.algorithm
    setproctitle.setproctitle(str_process_name)

    logging_config(args=cfg, process_id=process_id)

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                ", host name = " + hostname + "########" +
                ", process ID = " + str(os.getpid()) +
                ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0 and cfg.wandb_record:
        wandb.init(
            entity=cfg.entity,
            project=cfg.project,
            name=cfg.algorithm + " (d)" + str(cfg.partition_method) + "-" +str(cfg.dataset)+
                "-r" + str(cfg.comm_round) +
                "-e" + str(cfg.max_epochs) + "-" + str(cfg.model) + "-" +
                str(cfg.client_optimizer) + "-bs" + str(cfg.batch_size) +
                "-lr" + str(cfg.lr) + "-wd" + str(cfg.wd),
            config=dict(cfg)
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    logging.info("process_id = %d" % (process_id))

    device = torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")


    # load data
    if cfg.mode == "centralized":
        train_data_global, test_data_global, train_data_num, test_data_num, class_num, other_params \
            = load_data(load_as="training", args=cfg, process_id=process_id,
                        mode="centralized", task="centralized",
                        dataset=cfg.dataset, datadir=cfg.data_dir, batch_size=cfg.batch_size, num_workers=cfg.data_load_num_workers,
                        data_sampler=cfg.data_sampler,
                        resize=cfg.dataset_load_image_size, augmentation=cfg.dataset_aug)
    elif cfg.mode == "standalone":
        dataset = load_data(
                load_as="training", args=cfg, process_id=0, mode="standalone", task="federated", data_efficient_load=True,
                dirichlet_balance=False, dirichlet_min_p=None,
                dataset=cfg.dataset, datadir=cfg.data_dir,
                partition_method="hetero", partition_alpha=0.5,
                client_number=cfg.client_num_in_total, batch_size=cfg.batch_size, num_workers=cfg.data_load_num_workers,
                data_sampler=cfg.data_sampler,
                resize=cfg.dataset_load_image_size, augmentation=cfg.dataset_aug)

        [train_data_num, test_data_num, train_data_global, test_data_global,
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params] = dataset
    else:
        raise NotImplementedError
    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    # model = create_model(cfg, model_name=cfg.model, output_dim=dataset[7], **other_params)
    # model = create_model(cfg, model_name=cfg.model, output_dim=cfg.num_classes, **other_params)
    model = create_model(cfg, model_name=cfg.model, output_dim=cfg.model_output_dim,
                         pretrained=cfg.pretrained, device=device, **other_params)
    if cfg.pretrained:
        if cfg.model == "inceptionresnetv2":
            pass
        else:
            ckt = torch.load(cfg.pretrained_dir)
            if "model_state_dict" in ckt:
                if cfg.pretrained_submodel:
                    load_submodel(model_name=cfg.model, model=model,
                                pretrain_model_state_dict=ckt["model_state_dict"],
                                submodel_config=cfg.pretrained_layers)
                else:
                    model.load_state_dict(ckt["model_state_dict"])
            else:
                logging.info(f"ckt.keys: {list(ckt.keys())}")
                model.load_state_dict(ckt)


    # num_iterations = get_avg_num_iterations(train_data_local_num_dict, cfg.batch_size)
    global_num_iterations = train_data_num // cfg.batch_size
    num_iterations = global_num_iterations
    init_state_kargs = {}

    model_trainer = create_trainer(
        cfg, device, model, num_iterations=num_iterations,
        server_index=0, role='server',
        **init_state_kargs)

    metrics = Metrics(topks=[1], task=cfg.task)

    server_timer = ServerTimer(
        cfg,
        num_iterations,
        local_num_iterations_dict=None
    )
    train_tracker = RuntimeTracker(
        mode='Train',
        things_to_metric=metrics.metric_names,
        timer=server_timer,
        args=cfg
    )
    test_tracker = RuntimeTracker(
        mode='Test',
        things_to_metric=metrics.metric_names,
        timer=server_timer,
        args=cfg
    )
    # train_tracker = RuntimeTracker(things_to_track=metrics.metric_names)
    # test_tracker = RuntimeTracker(things_to_track=metrics.metric_names)

    # save_checkpoints_config = setup_checkpoint_config(cfg) if cfg.checkpoint_save else None
    save_checkpoints_config = setup_checkpoint_config(cfg)

    if cfg.model_dif_track:
        train_tracker.local_recorder.tracker_dict['model_dif_track'].set_initial_weight(
            model_trainer.get_model_params())

    if cfg.param_track:
        train_tracker.local_recorder.tracker_dict['param_track'].set_initial_weight(
            model_trainer.model)

    training_things_to_track = []
    if cfg.losses_track:
        training_things_to_track.append('losses_track')
    if cfg.grad_track:
        training_things_to_track.append('grad_track')
    if cfg.param_track_with_training:
        training_things_to_track.append('param_track')

    model_trainer.model = model_trainer.model.to(device)
    total_iterations = 0

    # print(model_trainer.flatter.samplers)
    # print(model_trainer.flatter.range_params)
    # print(model_trainer.flatter.range_params_name)

    # exit()

    for epoch in range(cfg.max_epochs):
        kwargs = {}
        kwargs["progress"] = epoch
        model_trainer.update_state(**kwargs)
        model_trainer.lr_schedule(epoch)
        if cfg.model_dif_track or (cfg.find_flat and "ServerCentric" in cfg.find_flat_FL):
            previous_model = copy.deepcopy(model_trainer.get_model_params())
            # kwargs["previous_model"] = previous_model
            kwargs["previous_model"] = None
        model_trainer.model = model_trainer.model.to(device)

        # for batch_idx in range(iterations):
        for batch_idx, train_batch_data in enumerate(train_data_global):
            # if cfg.level == 'DEBUG':
            # if batch_idx > 2:
            #     break
            total_iterations += 1
            # train_batch_data = get_train_batch_data()
            loss, output, labels = \
                model_trainer.train_one_step(
                    train_batch_data, device=device, args=cfg,
                    epoch=epoch, iteration=batch_idx,
                    tracker=train_tracker, metrics=metrics,
                    move_to_gpu=False, **kwargs)

            # if cfg.random_labels:
            #     logging.debug(f"train_data_global.dataset.labels: {labels[:10]}")

        save_checkpoint(cfg, save_checkpoints_config, extra_name="centralized",
                    epoch=server_timer.global_outer_epoch_idx,
                    model_state_dict=model_trainer.get_model_params(),
                    optimizer_state_dict={},
                    train_metric_info=train_tracker.get_metric_info(metrics),
                    test_metric_info=test_tracker.get_metric_info(metrics),
                    postfix=cfg.checkpoint_custom_name)

        if cfg.test:
            check_and_test(
                cfg,
                server_timer, model_trainer, test_data_global, device,
                train_tracker, test_tracker, metrics
            )

        server_timer.past_epochs(epochs=1)
        server_timer.past_comm_round(comm_round=1)

        # train_tracker.reset()
        # test_tracker.reset()





