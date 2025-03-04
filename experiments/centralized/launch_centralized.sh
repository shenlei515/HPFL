#!/bin/bash

######################################  standalone launch shell
cluster_name="${cluster_name:-localhost}"
model="${model:-resnet20}"
dataset="${dataset:-cifar10}"
algorithm="${algorithm:-FedAvg}"

dir_name=$(dirname "$PWD")

source ${dir_name}/configs_system/$cluster_name.conf
source ${dir_name}/configs_model/$model.conf
source ${dir_name}/configs_algorithm/$algorithm.conf
source ${dir_name}/main_args.conf

main_args="${main_args:-  }"

export WANDB_CONSOLE=off


$PYTHON ./main.py $main_args \









