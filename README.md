# Hot Plgguable Federated Learning

Official code for paper "Hot-pluggable Federated Learning: Bridging General and Personalized FL via Dynamic Selection" (ICLR 2025).


![](./Modular_market_pipeline_final4.pdf)

## Code Structure.

```
.
├── algorithms_standalone    # multiple round training and testing for algorithms
├── configs    # some default configurations
├── data_preprocessing    # loading datasets
├── experiments
│   ├── main_args.conf    # includes the default hyper-parameters.
│   ├── configs_algorithm    # includes the configuration hyper-parameters of the specific algorithms.
│   ├── configs_system    # includes the configuration hyper-parameters related to the running environment. This is designed for users' convenience of no need to specifying Python Path and Data Dir when launching experiments every time.
│   └── standalone    # includes scripts for HPFL and all algorithms, together with some default configuration.
├── requirements.yml    # dependency requirements
├── trainers
│   └── normal_trainer.py    # single round training for specific algorithms, like FedRoD, FedTHE, FedSAM, etc.
└── utils    # implementation of small algorithm components like CCA, CKA, MMD
```


## Launch Experiments.

1. Download and specific the data directory with parameter ``data_dir`` in command (For most dataset, the downloading will begin when runnning the script)

2. To launch HPFL and baselines, you can ``cd experiments/standlone``. And type ``./auto_pipeline.sh`` to run the code for 10 clients, ``./auto_pipeline_100client.sh`` for 100 clients. You can type ``dataset=cifar10 partition_alpha=0.1 partition_method=hetero algorithm=FedSAM ./launch_standalone.sh`` to set the dataset as CIFAR-10 and the a=0.1 of  partition method, while for FedSAM algorithm. 


## Wandb Usage

If you do not want to use wandb for recording, you can add ``wandb_record=False`` in to the command (such as in experiments/standalone/auto_pipeline.sh and experiments/standalone/auto_pipeline_100client.sh). Then you don't need to install and config wandb.

If else, please replace WANDB_KEY_EXAMPLE in configs/default.py, standalone/auto_pipeline.sh and experiments/standalone/auto_pipeline_100client.sh













