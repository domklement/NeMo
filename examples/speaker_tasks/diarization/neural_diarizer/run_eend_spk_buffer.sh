#!/bin/bash

# export WANDB_API_KEY="029a7cc91a821854880a156d05653e9b79d9e7d6"
export WANDB_MODE="online"

# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO

# manifest_path="/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/misc/manifests"
manifest_path="/tmp/dicow_data/nemo_manifests"
tokenizers_path="/tmp/tokenizers"

python /home/jovyan/NeMo/examples/speaker_tasks/diarization/neural_diarizer/eend_diar_train.py \
    --config-path=/home/jovyan/NeMo/examples/speaker_tasks/diarization/conf/neural_diarizer \
    --config-name=eend_spk_buff.yaml \
    model.train_ds.manifest_filepath=/home/jovyan/data/diar_data/compound_jiangyu_dataset/data_ssd/dev/ami_only_nemo_manifest_110s.jsonl \
    model.validation_ds.manifest_filepath=/home/jovyan/data/diar_data/compound_jiangyu_dataset/data_ssd/dev/ami_only_nemo_manifest_110s.jsonl \
    +model.train_ds.equalize_recording_lengths=True \
    trainer.devices=-1 \
    trainer.accelerator=gpu \
    trainer.strategy=ddp \
    trainer.max_epochs=1000 \
    trainer.log_every_n_steps=10 \
    +trainer.check_val_every_n_epoch=10 \
    exp_manager.checkpoint_callback_params.every_n_epochs=10 \
    +exp_manager.checkpoint_callback_params.save_last=False \
    model.train_ds.batch_size=64 \
    trainer.accumulate_grad_batches=1 \
    model.train_ds.session_len_sec=110 \
    model.train_ds.num_workers=10 \
    model.optim.sched.warmup_steps=2500 \
    model.optim.sched.min_lr=1e-07 \
    model.optim.lr=5e-5 \
    model.sortformer_modules.dropout_rate=0.2 \
    model.validation_ds.batch_size=4 \
    +model.encoder.prepend_global_tokens=true \
    +model.encoder.use_ce_spk_buffer=true \
    model.encoder.n_layers=8 \
    model.train_ds.session_len_sec=110 \
    model.encoder.att_context_size=[128,128] \
    model.encoder.self_attention_model=rel_pos_local_attn \
    model.encoder.att_context_style=regular \
    +model.encoder.global_tokens=1 \
    +model.encoder.global_tokens_spacing=1 \
    +model.encoder.global_attn_separate=false \
    +evaluate_at_start=False \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.exp_dir=/tmp/nemo_experiments/dk_nemo_eend_ami_overfit \
    exp_manager.create_wandb_logger=True \
    +trainer.gradient_clip_val=1.0 \
    exp_manager.wandb_logger_kwargs.name=110s_train_110s_valid_noforcefirstactive_localattn_spkbuffer_8l_1gpu_fixed \
    exp_manager.wandb_logger_kwargs.project=dk_nemo_eend_ami_overfit \
    name=110s_train_110s_valid_noforcefirstactive_localattn_spkbuffer_8l_1gpu_fixed
