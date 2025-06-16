#!/bin/bash

# export WANDB_API_KEY="029a7cc91a821854880a156d05653e9b79d9e7d6"
export WANDB_MODE="disabled"

# manifest_path="/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/misc/manifests"
manifest_path="/tmp/dicow_data/nemo_manifests"
tokenizers_path="/tmp/tokenizers"


python /home/jovyan/NeMo/examples/asr/ts_asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=/home/jovyan/NeMo/examples/asr/conf/fastconformer/hybrid_transducer_ctc \
    --config-name=fastconformer_hybrid_tdt_ctc_bpe_stno_parakeet_v2_0.6b.yaml \
    model.train_ds.manifest_filepath=/tmp/dicow_data/nemo_manifests/ami_nsf_train_sc_cutset_30s.jsonl \
    model.validation_ds.manifest_filepath=/tmp/dicow_data/nemo_manifests/notsofar_eval_sc_cutset.jsonl \
    model.tokenizer.dir=/tmp/tokenizers/ls960/tokenizer_spe_bpe_v500 \
    model.tokenizer.type=bpe \
    trainer.devices=-1 \
    trainer.accelerator=gpu \
    trainer.strategy=ddp \
    +evaluate_at_start=true \
    +decode_only=true \
    +init_from_ptl_ckpt=/home/jovyan/NeMo/misc/runs/parakeet_v2_0.6b_nsf_test_fddt_lr_mul_100_lr0.5_dicow_data_setup_nsf_dev_longform_2k_warmup_v3/checkpoints/parakeet_v2_0.6b_nsf_test_fddt_lr_mul_100_lr0.5_dicow_data_setup_nsf_dev_longform_2k_warmup_v3--val/best.ckpt \
    trainer.max_epochs=1000 \
    model.fddt_lr_multiplier=100 \
    trainer.log_every_n_steps=10 \
    model.train_ds.batch_size=16 \
    model.optim.sched.warmup_steps=2000 \
    model.optim.lr=0.5 \
    trainer.accumulate_grad_batches=4 \
    model.validation_ds.batch_size=1 \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.exp_dir=/tmp/nemo_experiments \
    exp_manager.checkpoint_callback_params.monitor=val/cp_wer \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=decoding \
    exp_manager.wandb_logger_kwargs.project=dk_nemo_tests \
    +init_from_pretrained=nvidia/parakeet-tdt-0.6b-v2 \
    name=decoding \
    +model.decoding.compute_timestamps=True \
    +model.decoding.preserve_alignments=False