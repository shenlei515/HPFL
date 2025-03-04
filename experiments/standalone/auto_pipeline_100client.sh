# # --------------------------------------- alpha=0.1, client=100 -------------------------------#

WANDB_API_KEY="WANDB_KEY_EXAMPLE"

# gpu_index=1 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=1 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&# gpu_index=1 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=1 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=0 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=$WANDB_API_KEY dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=4 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=4 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=4 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=1 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- alpha=0.05, client=100-------------------------------#
# gpu_index=1 comm_round=1000  partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000  partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000  partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=3 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=2 data_load_num_workers=0 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=10 wandb_key=$WANDB_API_KEY dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# gpu_index=1 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=2 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=2 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 CCA_component=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=2 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_ood bash base_hpfl.sh&

# gpu_index=3 data_load_num_workers=0 comm_round=1000 partition_alpha=0.05 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY dataset=Tiny-ImageNet-200 dataset_load_image_size=64 data_dir=./../../../data/tiny-imagenet-200 model_input_channels=3 num_classes=200 model_output_dim=200 plug_choose=plug_no_ood bash base_hpfl.sh&

# # --------------------------------------- noise_coefficient-------------------------------#
# gpu_index=1 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY noisy_coefficient=0 plug_choose=plug_no_ood bash base_hpfl.sh&
# gpu_index=1 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood noisy_coefficient=10 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood noisy_coefficient=100 bash base_hpfl.sh&
# gpu_index=4 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood noisy_coefficient=1000 bash base_hpfl.sh&
# gpu_index=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood noisy_coefficient=10000 bash base_hpfl.sh&

# gpu_index=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood noisy_coefficient=0 bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood noisy_coefficient=10 bash base_hpfl.sh&
# gpu_index=3 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood noisy_coefficient=100 bash base_hpfl.sh&
# gpu_index=5 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood noisy_coefficient=1000 bash base_hpfl.sh&
# gpu_index=5 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=fmnist dataset_load_image_size=28 data_dir=./../../../data model_input_channels=1 plug_choose=plug_no_ood noisy_coefficient=10000 bash base_hpfl.sh&

# gpu_index=6 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coefficient=0 bash base_hpfl.sh&
# gpu_index=6 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coefficient=10 bash base_hpfl.sh&
# gpu_index=5 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coefficient=100 bash base_hpfl.sh&
# gpu_index=5 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coefficient=1000 bash base_hpfl.sh&
# gpu_index=5 comm_round=1000 client_num_in_total=100 client_num_per_round=10 wandb_key=$WANDB_API_KEY dataset=cifar100 dataset_load_image_size=32 data_dir=./../../../data model_input_channels=3 num_classes=100 model_output_dim=100 plug_choose=plug_no_ood noisy_coefficient=10000 bash base_hpfl.sh&

# # --------------------------------------- reduce Plug-ins -------------------------------#
# gpu_index=0 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood reduce_plug=True reduce_threshold=0.03 bash base_hpfl.sh&
# gpu_index=0 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood reduce_plug=True reduce_threshold=0.025 bash base_hpfl.sh&
# gpu_index=1 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood reduce_plug=True reduce_threshold=0.015 bash base_hpfl.sh&
# gpu_index=3 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood reduce_plug=True reduce_threshold=0.01 bash base_hpfl.sh&
# gpu_index=2 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood reduce_plug=True reduce_threshold=0.035 bash base_hpfl.sh&
# gpu_index=2 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood reduce_plug=True reduce_threshold=0.04 bash base_hpfl.sh&
# gpu_index=3 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood reduce_plug=True reduce_threshold=0.045 bash base_hpfl.sh&
gpu_index=0 data_load_num_workers=0 comm_round=1000 client_num_in_total=100 client_num_per_round=10 HPFL_local_iteration=1 wandb_key=$WANDB_API_KEY plug_choose=plug_no_ood reduce_plug=True reduce_threshold=0.02 bash base_hpfl.sh&