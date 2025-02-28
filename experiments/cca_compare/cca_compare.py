import argparse
import logging
import os
import random
import socket
import sys
import yaml
import pickle
import traceback

import matplotlib.pyplot as plt
import matplotlib
import platform
sysstr = platform.system()
if ("Windows" in sysstr):
    matplotlib.use("TkAgg")
    logging.info ("On Windows, matplotlib use TkAgg")
else:
    matplotlib.use("Agg")
    logging.info ("On Linux, matplotlib use Agg")

import numpy as np
import psutil
import setproctitle
import torch
import wandb


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# from data_preprocessing.build import load_centralized_data
from model.build import create_model

from utils.get_activations import (
    register_get_activation_hooks, 
    get_dataset_activation, 
    load_dataset_activation,
    setup_save_activation_path
)
from utils.corr_methods import two_same_models_compare

from utils.data_utils import (
    get_local_num_iterations,
    get_avg_num_iterations
)

from utils.checkpoint import save_checkpoint, setup_checkpoint_config, setup_save_checkpoint_path


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--config_file", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str,
                        help="specify add which type of config")
    parser.add_argument("--extra_name", default=None, type=str,
                        help="specify extra name of checkpoint")
    parser.add_argument("--cca_reshape_method", default="Subsampling", type=str,
                        help="specify reshape method")
    parser.add_argument("--metric_thing", default="Acc1", type=str,
                        help="specify extra metric thing")
    parser.add_argument("--execute", default="cca_compare", type=str,
                        help="specify operation")
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def cal_activations(cfg, extra_name):

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic =True

    device = torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")

    # create model.
    if cfg.dataset in ["cifar10", "mnist", "cifar100"]:
        class_num = 10
    else:
        raise NotImplementedError 
    model = create_model(args=cfg, model_name=cfg.model, output_dim=class_num)
    # try:
    #     sample = (3, 224, 224)
    #     summary(model, sample)
    # except:
    #     logging.info("Scan model failed...")

    model_paras = model.named_parameters()
    print("===========================")
    for name, para in model_paras:
        print(name, para.shape)

    save_checkpoints_config = setup_checkpoint_config(cfg) if cfg.checkpoint_save else None

    diff_epoch_activation_dict = {}
    for epoch in save_checkpoints_config["checkpoint_epoch_list"]:
        logging.info("Getting epoch %s activations..." % (epoch))
        save_activation_path = setup_save_activation_path(
            save_checkpoints_config, 
            corr_dataset_len=cfg.corr_dataset_len,
            extra_name=extra_name, 
            epoch=epoch
        )

        save_checkpoint_path = setup_save_checkpoint_path(
            save_checkpoints_config, extra_name=extra_name, epoch=epoch)
        checkpoint = torch.load(save_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        activation, hook_handle_dict = register_get_activation_hooks(model, cfg.corr_layers_list)

        trainloader, testloader, train_data_num, test_data_num, class_num, other_params \
            = load_centralized_data(cfg, cfg.dataset, max_train_len=cfg.corr_dataset_len)
        activation_dataset = get_dataset_activation(trainloader, model, activation, device,
                                save_activation_path=save_activation_path)
        diff_epoch_activation_dict[epoch] = activation_dataset

        for _, hook_handle in hook_handle_dict.items():
            hook_handle.remove()





def cca_compare(cfg, extra_name, cca_reshape_method):

    save_checkpoints_config = setup_checkpoint_config(cfg) if cfg.checkpoint_save else None

    cca_dict = {}
    activation_file_X_path = setup_save_activation_path(
        save_checkpoints_config,
        corr_dataset_len=cfg.corr_dataset_len,
        extra_name=extra_name,
        epoch=save_checkpoints_config["checkpoint_epoch_list"][-1]
    )

    for epoch in save_checkpoints_config["checkpoint_epoch_list"]:
        activation_file_Y_path = setup_save_activation_path(
            save_checkpoints_config,
            corr_dataset_len=cfg.corr_dataset_len,
            extra_name=extra_name,
            epoch=epoch
        )
        

        activation_X = load_dataset_activation(activation_file_X_path, cfg.corr_layers_list)
        activation_Y = load_dataset_activation(activation_file_Y_path, cfg.corr_layers_list)
        logging.info("loading...  activations X, keys: {}, activations Y, keys: {}".format(
            activation_X.keys(), activation_Y.keys()
        ))

        cca = two_same_models_compare(cfg.corr_dataset_len, cfg.corr_layers_list,
                    activation_X, activation_Y, model_type="image", reshape_method=cca_reshape_method)
        cca_dict["corrs" + activation_file_X_path + activation_file_Y_path] = cca

        cca.write_correlations(
            "./cca_compare/corrs/" + cca_reshape_method + "-" + os.path.basename(activation_file_X_path) + \
                os.path.basename(activation_file_Y_path) + ".pickle")
        logging.info("Caculated {} epoch {} to epoch {} cca similarity".format(
            activation_file_Y_path, save_checkpoints_config["checkpoint_epoch_list"][-1], epoch
        ))


def cal_and_cca(cfg, extra_name, cca_reshape_method):
    cal_activations(cfg, extra_name)
    cca_compare(cfg, extra_name, cca_reshape_method)
