import argparse
import copy
import logging
import os
import random
import socket
import sys
import yaml

import traceback

import numpy as np
import psutil
import setproctitle
import torch
import wandb
# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


from utils.logger import (
    logging_config
)

from configs import get_cfg

from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
from algorithms_standalone.fednova.FedNovaManager import FedNovaManager
from algorithms_standalone.fedper.FedPERManager import FedPERManager
from algorithms_standalone.fedrod.FedRODManager import FEDRODManager
from algorithms_standalone.fedrep.FedREPManager import FEDREPManager
from algorithms_standalone.fedthe.FedTHEManager import FEDTHEManager
from algorithms_standalone.fedsam.FedSAMManager import FEDSAMManager
from utils.set_seed import set_seed
torch.multiprocessing.set_sharing_strategy('file_system')

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
    # initialize distributed computing (MPI)

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    #### set up cfg ####
    # default cfg
    cfg = get_cfg()
    print("Arguments configuration:::::\n",cfg)
    # add registered cfg
    # some arguments that are needed by build_config come from args.
    cfg.setup(args)

    # Build config once again
    cfg.setup(args)
    cfg.mode = 'standalone'

    # cfg.rank = process_id
    # if cfg.algorithm in ['FedAvg', 'AFedAvg', 'PSGD', 'APSGD', 'Local_PSGD']:
    #     cfg.client_index = process_id - 1
    # elif cfg.algorithm in ['DPSGD', 'DCD_PSGD', 'CHOCO_SGD', 'SAPS_FL']:
    #     cfg.client_index = process_id
    # else:
    #     raise NotImplementedError
    cfg.server_index = -1
    cfg.client_index = -1
    seed = cfg.seed
    print(!!!!!!seed)
    # seed = 99
    process_id = 0
    # show ultimate config
    logging.info(dict(cfg))

    # customize the process name
    str_process_name = cfg.algorithm + " (standalone):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    logging_config(args=cfg, process_id=process_id)

    # logging.info("In Fed Con model construction, model is {}, {}, {}".format(
    #     cfg.model, type(cfg.model), cfg.model == 'simple-cnn'
    # ))

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                ", host name = " + hostname + "########" +
                ", process ID = " + str(os.getpid()) +
                ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        if cfg.wandb_record:
            if cfg.wandb_key is not None:
                wandb.login(key=cfg.wandb_key)
            wandb.init(
                entity=cfg.entity,
                project=cfg.project,
                name=cfg.algorithm + " (d)" + str(cfg.partition_method) + "-" +str(cfg.dataset)+
                    "-r" + str(cfg.comm_round) +
                    "-e" + str(cfg.max_epochs) + "-" + str(cfg.model) + "-" +
                    str(cfg.client_optimizer) + "-bs" + str(cfg.batch_size) +
                    "-lr" + str(cfg.lr) + "-wd" + str(cfg.wd),
                config=dict(cfg),
                settings=wandb.Settings(start_method="fork")
            )
        if cfg.wandb_offline:
            os.environ['WANDB_MODE'] = 'dryrun'

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    set_seed(seed)

    device = torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")
    print("cfg.algorithm===========", cfg.algorithm)
    print("cfg.fedavg_cfl", cfg.fedavg_cfl)
    if cfg.fedavg_cfl:
        fedavg_manager = FedAVGManager(device, cfg)
        print("entering fedavg_manager.cfl_train()")
        fedavg_manager.cfl_train()
        # fedavg_manager.generate_personalized_data()
        # print("entering fedavg_manager.cfl_test()")
        fedavg_manager.cfl_test()
    else:
        if cfg.algorithm == 'FedAvg':
            fedavg_manager = FedAVGManager(device, cfg)
            fedavg_manager.train() # remember to recover when the CCA is checked
        elif cfg.algorithm == 'FedNova':
            fednova_manager = FedNovaManager(device, cfg)
            fednova_manager.train()
        elif cfg.algorithm == 'FedPer':
            fedper_manager=FedPERManager(device, cfg)
            fedper_manager.train()
            set_seed(seed)
            fedper_manager.generate_personalized_data()
            set_seed(seed)
            fedper_manager.test_with_personalized_batch("index")
            set_seed(seed)
            fedper_manager.test_plug_globally()
            set_seed(seed)
        elif cfg.algorithm == 'FedRod':
            fedrod_manager=FEDRODManager(device, cfg)
            fedrod_manager.train()
            set_seed(seed)
            fedrod_manager.generate_personalized_data()
            set_seed(seed)
            fedrod_manager.test_with_personalized_batch("index")
            set_seed(seed)
            fedrod_manager.test_plug_globally()
            set_seed(seed)
        elif cfg.algorithm == 'FedRep':
            fedrep_manager=FEDREPManager(device, cfg)
            fedrep_manager.train()
            set_seed(seed)
            fedrep_manager.generate_personalized_data()
            set_seed(seed)
            fedrep_manager.test_with_personalized_batch("index")
            set_seed(seed)
            fedrep_manager.test_plug_globally()
            set_seed(seed)
        elif cfg.algorithm == 'FedTHE':
            fedthe_manager=FEDTHEManager(device, cfg)
            fedthe_manager.train()
            set_seed(seed)
            fedthe_manager.generate_personalized_data()
            set_seed(seed)
            fedthe_manager.test_with_personalized_batch("index")
            set_seed(seed)
            fedthe_manager.test_plug_globally()
            set_seed(seed)
        elif cfg.algorithm == 'FedSAM':
            fedthe_manager=FEDSAMManager(device, cfg)
            fedthe_manager.train()
            set_seed(seed)
            fedthe_manager.generate_personalized_data()
            set_seed(seed)
            fedthe_manager.test_with_personalized_batch("index")
            set_seed(seed)
            fedthe_manager.test_plug_globally()
            set_seed(seed)
        else:
            raise NotImplementedError
    
    
    if cfg.HPFL:
        fedavg_manager.check_and_test()
        set_seed(seed)
        if not cfg.if_OOD:
            # if cfg.select_with_distance:
            #     fedavg_manager.test_all_samples("distance")
            # print("after test_all_samples::::::::::::::::", list(fedavg_manager.client_list[0].trainer.model.named_parameters()))
            # set_seed(seed)
            
            # fedavg_manager.test_ensemble("Avg_all")
            # print("after test_ensemble(\"Avg_all\")::::::::::::::::", list(fedavg_manager.client_list[0].trainer.model.named_parameters()))
            # set_seed(seed)
            
            # fedavg_manager.test_all_pluggable()
            # print("after test_all_pluggable::::::::::::::::", list(fedavg_manager.client_list[0].trainer.model.named_parameters()))
            # set_seed(seed)
            # # fedavg_manager.test_locally()
            # # print("after test_locally::::::::::::::::", list(fedavg_manager.client_list[0].trainer.model.named_parameters()))
            # # set_seed(seed)
            # fedavg_manager.test_selection("CCA")
            # print("after test_selection(\"CCA\")::::::::::::::::", list(fedavg_manager.client_list[0].trainer.model.named_parameters()))
            # set_seed(seed)
            # if cfg.select_multi_Pluggable:
            #     fedavg_manager.test_ensemble("multi-Pluggable_with_CCA")
            #     print("after test_ensemble(\"multi-Pluggable_with_CCA\")::::::::::::::::", list(fedavg_manager.client_list[0].trainer.model.named_parameters()))
            # set_seed(seed)
            # # may change the trainer.model in clients
            # fedavg_manager.generate_non_iid_batch()
            # set_seed(seed)
            # if cfg.select_with_KL:
            #     fedavg_manager.test_with_selected_pluggable_for_clients("KL")
            #     set_seed(seed)
            #     fedavg_manager.test_with_selected_pluggable_for_batch("KL")
            #     set_seed(seed)
            # if cfg.select_with_CCA:
            #     fedavg_manager.test_with_selected_pluggable_for_clients("CCA")
            #     print("after test_with_selected_pluggable_for_clients(\"CCA\")::::::::::::::::", list(fedavg_manager.client_list[0].trainer.model.named_parameters()))
            #     set_seed(seed)
            #     fedavg_manager.test_with_selected_pluggable_for_batch("CCA")
            #     print("after test_with_selected_pluggable_for_batch(\"CCA\")::::::::::::::::", list(fedavg_manager.client_list[0].trainer.model.named_parameters()))
            #     set_seed(seed)
            # # fedavg_manager.test_with_personalized_batch("CCA")
            # # set_seed(seed)
            # # print("after test_with_personalized_batch(\"index\")::::::::::::::::", list(fedavg_manager.client_list[0].trainer.model.named_parameters()))
            fedavg_manager.generate_personalized_data()
            set_seed(seed)
            if not cfg.finetune_FedAvg:
                fedavg_manager.test_with_personalized_batch("index")
                set_seed(seed)
                
                # fedavg_manager.test_with_CCA_between_personalized_batch("CCA")
                # set_seed(seed)
                fedavg_manager.test_with_personalized_batch("MMD")
                set_seed(seed)
                
                # fedavg_manager.test_with_personalized_batch("CCA")
                # set_seed(seed)
                # fedavg_manager.test_with_personalized_batch("linear-CKA")
                # set_seed(seed)
                # fedavg_manager.test_with_personalized_batch("rbf-CKA")
                # set_seed(seed)
                # fedavg_manager.test_with_personalized_batch("linear-CKA_debias")
                # set_seed(seed)
                # fedavg_manager.test_with_personalized_batch("rbf-CKA_debias")
                # set_seed(seed)
                
                # for i in range(len(fedavg_manager.client_list)):
                #     fedavg_manager.test_with_personalized_batch_with_plugn(i)
                # set_seed(seed)
                # fedavg_manager.test_with_global_with_plugs()
                # set_seed(seed)
            else:
                fedavg_manager.test_with_personalized_batch("index")
                set_seed(seed)
        else:
            # fedavg_manager.test_all_samples("OOD")
            # set_seed(seed)
            # fedavg_manager.test_ensemble("Avg_all")
            # set_seed(seed)
            # fedavg_manager.test_ensemble("multi-Pluggable_with_OOD")
            # set_seed(seed)
            # fedavg_manager.test_selection("OOD")
            # set_seed(seed)
            # fedavg_manager.test_selection_on_train("OOD")
            # set_seed(seed)
            # fedavg_manager.test_ACC_of_OOD()
            # set_seed(seed)
            # if not cfg.OOD_feat_noise:
            #     fedavg_manager.feat_norm()
            #     set_seed(seed)
            fedavg_manager.generate_personalized_data()
            set_seed(seed)
            # for i in range(len(fedavg_manager.client_list)):
            #     fedavg_manager.test_with_personalized_batch_with_plugn(i)
            # set_seed(seed)
            fedavg_manager.test_with_personalized_batch("personalized_with_OOD")
            set_seed(seed)
            pass

        print("Pluggable_training_iteration", cfg.HPFL_local_iteration)
        print("unfreeze_layers", cfg.freeze_layer)
        print("centralized_training_epoch", cfg.central_trainning_epoch)
        print("global_epoch_in_FedAvg:", cfg.comm_round)
        print("local_epoch_in_FedAvg:", cfg.global_epochs_per_round)
        print("model_name:", cfg.model)

