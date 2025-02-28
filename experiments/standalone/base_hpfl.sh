#!/bin/bash

export entity="leishen"
export project="test"
export level=${level:-INFO}
export exp_mode=${exp_mode:-"ready"}

# export WANDB_MODE=offline

export plug_choose=${plug_choose:-plug_no_ood}

source hpfl_exp/$plug_choose.sh


export cluster_name=${cluster_name:-localhost}
export gpu_index=${gpu_index:-2}
export client_num_in_total=${client_num_in_total:-10}
export client_num_per_round=${client_num_per_round:-5}

export client_num_per_round=5
export momentum=0.0
export global_epochs_per_round=1
export max_epochs=${max_epochs:-1000}

export partition_alpha=${partition_alpha:-0.1}

export model=${model:-resnet18_v2}
# export model="resnet18_v2"
export lr=${lr:-0.1}
# export lr=0.01


# export sched="StepLR"
# export lr_decay_rate=0.992
export sched=${sched:-"no"}


# export partition_method='hetero'
# export partition_alpha=0.5
# export partition_method='long-tail'
# export partition_alpha=0.99
export partition_method=${partition_method:-'hetero'}
export partition_alpha=${partition_alpha:-0.1}



# export dirichlet_balance=True

# export batch_size=32
# export lr=0.04

export batch_size=${batch_size:-128}
export wd=${wd:-0.0001}
export HPFL_local_iteration=${HPFL_local_iteration:-10}



export record_dataframe=False
export wandb_save_record_dataframe=False
# export wandb_upload_client_list="[0,1,2,3,4]"
export wandb_upload_client_list="[]"




export script=${script:-'./launch_standalone.sh'}


bash ${script}












