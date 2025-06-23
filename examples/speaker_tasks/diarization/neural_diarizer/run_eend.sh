#!/bin/bash
e
# export WANDB_API_KEY="029a7cc91a821854880a156d05653e9b79d9e7d6"
export WANDB_MODE="online"

# manifest_path="/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/misc/manifests"
manifest_path="/tmp/dicow_data/nemo_manifests"
tokenizers_path="/tmp/tokenizers"

python /home/jovyan/NeMo/examples/speaker_tasks/diarization/neural_diarizer/eend_diar_train.py \
    --config-path=/home/jovyan/NeMo/examples/speaker_tasks/diarization/conf/neural_diarizer \
    --config-name=eend_pil_loss_4spk.yaml \
    model.train_ds.manifest_filepath=/tmp/diar_data/compound_jiangyu_dataset/data_ssd/train/nemo_manifest.json \
    model.validation_ds.manifest_filepath=/tmp/diar_data/compound_jiangyu_dataset/data_ssd/dev/nemo_manifest_300s.jsonl \
    +model.train_ds.equalize_recording_lengths=True \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.strategy=auto \
    trainer.max_epochs=1000 \
    trainer.log_every_n_steps=10 \
    +trainer.check_val_every_n_epoch=1 \
    model.train_ds.batch_size=16 \
    trainer.accumulate_grad_batches=2 \
    model.optim.sched.warmup_steps=2500 \
    model.optim.sched.min_lr=1e-07 \
    model.optim.lr=5e-5 \
    model.validation_ds.batch_size=16\
    +init_from_nest=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.exp_dir=/tmp/nemo_experiments \
    exp_manager.create_wandb_logger=True \
    +trainer.gradient_clip_val=5.0 \
    exp_manager.wandb_logger_kwargs.name=fastconformer_eend_18l_pil_8compound_equalized_32bs_lr_5e-5_warmup_2.5k_gradclip_5_10spks_90s_nest \
    exp_manager.wandb_logger_kwargs.project=dk_nemo_tests \
    name=fastconformer_eend_18l_pil_8compound_equalized_32bs_lr_5e-5_warmup_2.5k_gradclip_5_10spks_90s_nest
