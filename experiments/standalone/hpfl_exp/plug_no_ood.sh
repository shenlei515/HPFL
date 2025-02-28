export HPFL=${HPFL:-True}
export HPFL_local_iteration=${HPFL_local_iteration:-10}
export trainer_param_groups=${trainer_param_groups:-False}

export select_with_distance=True
export select_with_KL=False
export select_with_CCA=True

export select_multi_Pluggable=True
export num_Pluggable=${num_Pluggable:-5}

export freeze_layer=${freeze_layer:-1}
export central_trainning_epoch=${central_trainning_epoch:-0}
export feature_reduction_dim=${feature_reduction_dim:-64}
export non_iid_for_batch_data=${non_iid_for_batch_data:-True}
export CCA_sample=${CCA_sample:-128} # 512 // 4

export if_eval=${if_eval:-True}
export if_OOD=False
export OOD_independ_classifier=${OOD_independ_classifier:-True}
export OOD_feat_noise=${OOD_feat_noise:-True}
export OOD_noise_option=${OOD_noise_option:-random}
export freeze_OOD_backbone=${freeze_OOD_backbone:-True}
export CCA_component=${CCA_component:-5}