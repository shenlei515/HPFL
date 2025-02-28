#!/bin/bash

# --------------------------------------- alpha=0.1, client=10-------------------------------#
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=0 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=1 data_load_num_workers=0 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc CCA_component=1 seed=0 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=3 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=3 data_load_num_workers=0 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# --------------------------------------- alpha=0.05, client=10-------------------------------#
# gpu_index=3 comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=5 comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=5 comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=5 comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=6 comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=1 data_load_num_workers=0 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=6 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc CCA_component=1 seed=0 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=6 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=5 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=6 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=0 data_load_num_workers=0 comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedPer/FedRep/FedRod 10client alpha=0.1-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedRod HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedRod HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedRod HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=2 comm_round=1000 algorithm=FedPer HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedRep HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 algorithm=FedRod HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedPer/FedRep/FedRod 10client alpha=0.05-------------------------------#
# gpu_index=3 comm_round=1000 algorithm=FedPer partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=5 comm_round=1000 algorithm=FedRep partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=6 comm_round=1000 algorithm=FedRod partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=5 comm_round=1000 algorithm=FedPer partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=5 comm_round=1000 algorithm=FedRep partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 CCA_component=1 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=5 comm_round=1000 algorithm=FedRod partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=6 comm_round=1000 algorithm=FedPer partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=6 comm_round=1000 algorithm=FedRep partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=6 comm_round=1000 algorithm=FedRod partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 algorithm=FedPer partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedRep partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 algorithm=FedRod partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedPer/FedRep/FedRod 100client alpha=0.1-------------------------------#
# gpu_index=2 comm_round=1000 data_load_num_workers=0 algorithm=FedPer HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=3 comm_round=1000 data_load_num_workers=0 algorithm=FedRep HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=3 comm_round=1000 data_load_num_workers=0 algorithm=FedRod HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh

# gpu_index=2 comm_round=1000 data_load_num_workers=0 algorithm=FedPer HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=1 comm_round=1000 data_load_num_workers=0 algorithm=FedRep HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 CCA_component=1 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=2 comm_round=1000 data_load_num_workers=0 algorithm=FedRod HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh

# gpu_index=0 comm_round=1000 data_load_num_workers=0 algorithm=FedPer HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=0 comm_round=1000 data_load_num_workers=0 algorithm=FedRep HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=0 comm_round=1000 data_load_num_workers=0 algorithm=FedRod HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh

# gpu_index=1 comm_round=1000 data_load_num_workers=0 algorithm=FedPer HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 data_load_num_workers=0 algorithm=FedRep HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 data_load_num_workers=0 algorithm=FedRod HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedPer/FedRep/FedRod 100client alpha=0.05-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedPer partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 algorithm=FedRep partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedRod partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 algorithm=FedPer partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 algorithm=FedRep partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 CCA_component=1 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 algorithm=FedRod partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 algorithm=FedPer partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 algorithm=FedRep partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 algorithm=FedRod partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 data_load_num_workers=0 algorithm=FedPer partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 data_load_num_workers=0 algorithm=FedRep partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 data_load_num_workers=0 algorithm=FedRod partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedAvg finetuning -------------------------------#
# gpu_index=1 finetune_FedAvg=True comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=1 finetune_FedAvg=True HPFL_local_iteration=10 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=1 finetune_FedAvg=True comm_round=1000  partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=1 finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=10 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=1 finetune_FedAvg=True data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 finetune_FedAvg=True data_load_num_workers=0 comm_round=1000 HPFL_local_iteration=10 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 finetune_FedAvg=True data_load_num_workers=0 partition_alpha=0.05 comm_round=1000 HPFL_local_iteration=1 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 finetune_FedAvg=True data_load_num_workers=0 partition_alpha=0.05 comm_round=1000 HPFL_local_iteration=10 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=4 finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=4 finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=4 finetune_FedAvg=True comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=4 finetune_FedAvg=True comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=4 finetune_FedAvg=True data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 finetune_FedAvg=True data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 finetune_FedAvg=True data_load_num_workers=0 partition_alpha=0.05 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 finetune_FedAvg=True data_load_num_workers=0 partition_alpha=0.05 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=1 finetune_FedAvg=True comm_round=1000  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=2 finetune_FedAvg=True comm_round=1000  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=2 finetune_FedAvg=True comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=3 finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=10 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=4 finetune_FedAvg=True data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 finetune_FedAvg=True data_load_num_workers=0 HPFL_local_iteration=10 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 finetune_FedAvg=True data_load_num_workers=0 HPFL_local_iteration=1 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 finetune_FedAvg=True data_load_num_workers=0 HPFL_local_iteration=10 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=0 finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=1 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=10 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 finetune_FedAvg=True data_load_num_workers=0 HPFL_local_iteration=1 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 finetune_FedAvg=True data_load_num_workers=0 HPFL_local_iteration=10 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 finetune_FedAvg=True data_load_num_workers=0 HPFL_local_iteration=1 partition_alpha=0.05 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 finetune_FedAvg=True data_load_num_workers=0 HPFL_local_iteration=10 partition_alpha=0.05 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# --------------------------------------- different number of freezed layers -------------------------------#
# CIFAR-10 alpha=0.1 client_num=10
# gpu_index=0 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=1 bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=1 bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=3 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=3 bash base_hpfl.sh&

# gpu_index=2 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=4 bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=4 bash base_hpfl.sh&

# gpu_index=3 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=5 bash base_hpfl.sh&
gpu_index=2 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=5 bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=6 bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=6 bash base_hpfl.sh&

# CIFAR-10 alpha=0.05 client_num=10
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=3 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=10 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=3 bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=4 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=10 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=4 bash base_hpfl.sh&

# gpu_index=4 comm_round=1000 HPFL_local_iteration=1 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=5 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 HPFL_local_iteration=10 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=5 bash base_hpfl.sh&

# gpu_index=3 comm_round=1000 HPFL_local_iteration=1 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=6 bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 HPFL_local_iteration=10 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=6 bash base_hpfl.sh&
# CIFAR-10 alpha=0.1 client_num=100
# gpu_index=6 comm_round=1000 data_load_num_workers=0 HPFL_local_iteration=1 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=3 bash base_hpfl.sh&
# gpu_index=6 comm_round=1000 data_load_num_workers=0 HPFL_local_iteration=10 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=3 bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 HPFL_local_iteration=1 data_load_num_workers=0 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=4 bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 HPFL_local_iteration=10 data_load_num_workers=0 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=4 bash base_hpfl.sh&

# gpu_index=2 comm_round=1000 HPFL_local_iteration=1 data_load_num_workers=0 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=5 bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 HPFL_local_iteration=10 data_load_num_workers=0 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=5 bash base_hpfl.sh&

# gpu_index=3 comm_round=1000 HPFL_local_iteration=1 data_load_num_workers=0 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=6 bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 HPFL_local_iteration=10 data_load_num_workers=0 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood freeze_layer=6 bash base_hpfl.sh&

# --------------------------------------- different level of noisy coefficient -------------------------------#

# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc noisy_coefficient=0 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coefficient=10 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coefficient=100 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coefficient=1000 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coefficient=10000 bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood noisy_coefficient=0 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood noisy_coefficient=10 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood noisy_coefficient=100 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood noisy_coefficient=1000 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood noisy_coefficient=10000 bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coefficient=0 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coefficient=10 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coefficient=100 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coefficient=1000 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coefficient=10000 bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=0 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=10 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=100 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=1000 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=10000 bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=0 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=10 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=100 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=1000 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=10000 bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=0 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=10 bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=100 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=1000 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 partition_alpha=0.05 plug_choose=plug_no_ood noisy_coefficient=10000 bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc noise_type=random noisy_coefficient=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc noise_type=random noisy_coefficient=10 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc noise_type=random noisy_coefficient=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc noise_type=random noisy_coefficient=1000 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc noise_type=random noisy_coefficient=10000 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc noisy_coefficient=10 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc noisy_coefficient=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc noisy_coefficient=1000 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc noisy_coefficient=10000 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coefficient=10 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coefficient=100 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coefficient=1000 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coefficient=10000 bash base_hpfl.sh&

# gpu_index=3 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coeffiency=10 bash base_hpfl.sh
# gpu_index=3 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coeffiency=10 bash base_hpfl.sh
# gpu_index=3 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coeffiency=10 bash base_hpfl.sh
# gpu_index=3 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coeffiency=10 bash base_hpfl.sh

# gpu_index=4 comm_round=1000 HPFL_local_iteration=100 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coeffiency=10 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 HPFL_local_iteration=100 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coeffiency=100 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 HPFL_local_iteration=100 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coeffiency=1000 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 HPFL_local_iteration=100 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood noisy_coeffiency=10000 bash base_hpfl.sh&

# gpu_index=3 comm_round=1000  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=3 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=3 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=3 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh

# --------------------------------------- FedRod + HPFL -------------------------------#
# gpu_index=0 load_backbone_from=FedRod comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 load_backbone_from=FedRod comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 load_backbone_from=FedRod comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 load_backbone_from=FedRod data_load_num_workers=0 comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# --------------------------------------- Federated Continous Learning -------------------------------#
# gpu_index=3 comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc fedavg_cfl=True fcl_freeze_backbone=False lr=0.05 HPFL=False bash base_hpfl.sh

# --------------------------------------- simple-CNN -------------------------------#
# gpu_index=1 model=simple-cnn comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 model=simple-cnn comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=1 model=simple-cnn-mnist comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=1 model=simple-cnn comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 model=simple-cnn comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=1 model=simple-cnn-mnist comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=1 model=simple-cnn comm_round=1000 algorithm=FedTHE FedTHE_finetune=False HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=2 model=simple-cnn comm_round=1000 algorithm=FedPer HPFL=False HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 model=simple-cnn comm_round=1000 algorithm=FedPer HPFL=False HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 model=simple-cnn-mnist comm_round=1000 algorithm=FedPer HPFL=False HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh

# gpu_index=3 model=simple-cnn comm_round=1000 algorithm=FedRep HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 model=simple-cnn comm_round=1000 algorithm=FedRep HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 model=simple-cnn-mnist comm_round=1000 algorithm=FedRep HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=0 model=simple-cnn comm_round=1000 algorithm=FedRod HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 model=simple-cnn comm_round=1000 algorithm=FedRod HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 model=simple-cnn-mnist comm_round=1000 algorithm=FedRod HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=0 model=simple-cnn finetune_FedAvg=True comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 model=simple-cnn finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=2 model=simple-cnn-mnist finetune_FedAvg=True comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=3 model=simple-cnn finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 model=simple-cnn finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh
# gpu_index=3 model=simple-cnn-mnist finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh

# --------------------------------------- MobileNet v1 -------------------------------#
# gpu_index=0 model=mobilenet comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 model=mobilenet comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 model=mobilenet comm_round=1000 algorithm=FedPer HPFL=False HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 model=mobilenet comm_round=1000 algorithm=FedRep HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 model=mobilenet comm_round=1000 algorithm=FedRod HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 model=mobilenet comm_round=1000 algorithm=FedTHE FedTHE_finetune=False HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 model=mobilenet finetune_FedAvg=True comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 model=mobilenet finetune_FedAvg=True comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&

# --------------------------------------- group_feature noisy_coefficient=1 -------------------------------#
# gpu_index=1 group_feat=True comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 group_feat=True comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=2 group_feat=True comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 group_feat=True comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=3 group_feat=True data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 group_feat=True data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=0 group_feat=True comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 group_feat=True comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&

# --------------------------------------- group_feature noisy_coefficient=0 -------------------------------#
# gpu_index=1 noisy_coefficient=0 group_feat=True comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 noisy_coefficient=0 group_feat=True comm_round=1000 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 noisy_coefficient=0 group_feat=True comm_round=1000 partition_alpha=0.05 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 noisy_coefficient=0 group_feat=True comm_round=1000 partition_alpha=0.05 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 noisy_coefficient=0 group_feat=True data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 noisy_coefficient=0 group_feat=True data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 noisy_coefficient=0 group_feat=True comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 noisy_coefficient=0 group_feat=True comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&

# --------------------------------------- FedTHE+ -------------------------------#

# # --------------------------------------- FedTHE+ 10client alpha=0.1-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedTHE HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedTHE HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedTHE HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 algorithm=FedTHE HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedTHE+ 10client alpha=0.05-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedTHE partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedTHE partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedTHE partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 algorithm=FedTHE partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedTHE+ 100client alpha=0.1-------------------------------#
# gpu_index=0 comm_round=1000 data_load_num_workers=0 algorithm=FedTHE HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 data_load_num_workers=0 algorithm=FedTHE HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 data_load_num_workers=0 algorithm=FedTHE HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 data_load_num_workers=0 algorithm=FedTHE HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # ---------------------------------------FedTHE+ 100client alpha=0.05-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedTHE partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedTHE partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedTHE partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 data_load_num_workers=0 algorithm=FedTHE partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# --------------------------------------- FedTHE -------------------------------#

# --------------------------------------- FedTHE 10client alpha=0.1-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedTHE 10client alpha=0.05-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedTHE 100client alpha=0.1-------------------------------#
# gpu_index=0 comm_round=1000 FedTHE_finetune=False data_load_num_workers=0 algorithm=FedTHE HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 FedTHE_finetune=False data_load_num_workers=0 algorithm=FedTHE HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 FedTHE_finetune=False data_load_num_workers=0 algorithm=FedTHE HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 FedTHE_finetune=False data_load_num_workers=0 algorithm=FedTHE HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # ---------------------------------------FedTHE 100client alpha=0.05-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedTHE FedTHE_finetune=False partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 data_load_num_workers=0 algorithm=FedTHE FedTHE_finetune=False partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # ---------------------------------------Rerun FedPer-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedPer partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=2 lr=0.01 comm_round=1000 algorithm=FedPer HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&$
# gpu_index=1 lr=0.01 comm_round=1000 algorithm=FedPer partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 lr=0.01 model=simple-cnn comm_round=1000 algorithm=FedRep HPFL=False wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&


# --------------------------------------- FedSAM -------------------------------#
# --------------------------------------- HPFL_local_iteration=10 -------------------------------#
# --------------------------------------- FedSAM 10client alpha=0.1-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedSAM HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedSAM HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedSAM HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 algorithm=FedSAM HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedSAM 10client alpha=0.05-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False  HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedSAM 100client alpha=0.1-------------------------------#
# gpu_index=0 comm_round=1000 data_load_num_workers=0 algorithm=FedSAM HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 data_load_num_workers=0 algorithm=FedSAM HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 data_load_num_workers=0 algorithm=FedSAM HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 data_load_num_workers=0 algorithm=FedSAM HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # ---------------------------------------FedSAM 100client alpha=0.05-------------------------------#
# gpu_index=0 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 data_load_num_workers=0 algorithm=FedSAM partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# --------------------------------------- HPFL_local_iteration=1 -------------------------------#
# --------------------------------------- FedSAM 10client alpha=0.1-------------------------------#
# gpu_index=0 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedSAM 10client alpha=0.05-------------------------------#
# gpu_index=0 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False  wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- FedSAM 100client alpha=0.1-------------------------------#
# gpu_index=0 HPFL_local_iteration=1 comm_round=1000 data_load_num_workers=0 algorithm=FedSAM HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 HPFL_local_iteration=1 comm_round=1000 data_load_num_workers=0 algorithm=FedSAM HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 HPFL_local_iteration=1 comm_round=1000 data_load_num_workers=0 algorithm=FedSAM HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 HPFL_local_iteration=1 comm_round=1000 data_load_num_workers=0 algorithm=FedSAM HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # ---------------------------------------FedSAM 100client alpha=0.05-------------------------------#
# gpu_index=0 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 HPFL_local_iteration=1 comm_round=1000 algorithm=FedSAM partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 HPFL_local_iteration=1 comm_round=1000 data_load_num_workers=0 algorithm=FedSAM partition_alpha=0.05 HPFL=False client_num_in_total=100 client_num_per_round=10 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# --------------------------------------- Different Seeds -------------------------------#
gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_ood bash base_hpfl.sh&
gpu_index=1 comm_round=1000 wandb_key=52b3fac9fa69e6ffc4a0160f881c732eca640dbc plug_choose=plug_ood HPFL_local_iteration=10 bash base_hpfl.sh&