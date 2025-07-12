#!/bin/bash

# export WANDB_API_KEY="029a7cc91a821854880a156d05653e9b79d9e7d6"
export WANDB_MODE="online"

# manifest_path="/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/misc/manifests"
manifest_path="/tmp/dicow_data/nemo_manifests"
tokenizers_path="/tmp/tokenizers"

python /home/jovyan/NeMo/examples/speaker_tasks/diarization/neural_diarizer/sortformer_diar_train.py \
    --config-path=/home/jovyan/NeMo/examples/speaker_tasks/diarization/conf/neural_diarizer \
    --config-name=sortformer_diarizer_hybrid_loss_4spk-v1.yaml \
    model.train_ds.manifest_filepath=/home/jovyan/data/diar_data/compound_jiangyu_dataset/data_ssd/dev/ami_only_nemo_manifest_110s.jsonl \
    model.validation_ds.manifest_filepath=/home/jovyan/data/diar_data/compound_jiangyu_dataset/data_ssd/dev/ami_only_nemo_manifest_110s.jsonl \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.strategy=auto \
    trainer.max_epochs=100 \
    model.train_ds.session_len_sec=110 \
    model.validation_ds.session_len_sec=110 \
    trainer.log_every_n_steps=10 \
    model.validation_ds.batch_size=1 \
    model.encoder.n_layers=18 \
    +init_from_nest=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.exp_dir=/tmp/nemo_experiments \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=sortformer_test_4spks_90schunked \
    exp_manager.wandb_logger_kwargs.project=dk_nemo_tests \
    name=sortformer_test_4spks_90schunked

# python /home/jovyan/NeMo/examples/speaker_tasks/diarization/neural_diarizer/sortformer_diar_train.py \
#     --config-path=/home/jovyan/NeMo/examples/speaker_tasks/diarization/conf/neural_diarizer \
#     --config-name=sortformer_diarizer_hybrid_loss_4spk-v1.yaml \
#     model.train_ds.manifest_filepath=/home/jovyan/data/diar_data/compound_jiangyu_dataset/data_ssd/dev/ami_only_nemo_manifest_110s.jsonl \
#     model.validation_ds.manifest_filepath=/home/jovyan/data/diar_data/compound_jiangyu_dataset/data_ssd/dev/ami_only_nemo_manifest_110s.jsonl \
#     trainer.devices=1 \
#     trainer.accelerator=gpu \
#     trainer.strategy=auto \
#     trainer.max_epochs=100 \
#     trainer.log_every_n_steps=10 \
#     trainer.val_check_interval=0.5 \
#     model.validation_ds.batch_size=1 \
#     exp_manager.resume_ignore_no_checkpoint=True \
#     exp_manager.exp_dir=/tmp/nemo_experiments \
#     exp_manager.create_wandb_logger=True \
#     exp_manager.wandb_logger_kwargs.name=sortformer_test_4spks_90schunked \
#     exp_manager.wandb_logger_kwargs.project=dk_nemo_tests \
#     name=sortformer_test_4spks_90schunked