#!/bin/bash

# export WANDB_API_KEY="029a7cc91a821854880a156d05653e9b79d9e7d6"
export WANDB_MODE="online"

# manifest_path="/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/misc/manifests"
manifest_path="/tmp/dicow_data/nemo_manifests"
tokenizers_path="/tmp/tokenizers"

python /storage/brno12-cerit/home/dklement/speech/ASR/NeMo/examples/asr/ts_asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path="/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/examples/asr/conf/fastconformer/hybrid_transducer_ctc" \
    --config-name="fastconformer_hybrid_tdt_ctc_bpe_stno_parakeet_v2_0.6b.yaml" \
    model.train_ds.manifest_filepath=/storage/brno12-cerit/home/dklement/speech/chime9_mcorec/manifests/mcorec_train_trimmed_sg_nemo_manifest.jsonl \
    model.validation_ds.manifest_filepath=/storage/brno12-cerit/home/dklement/speech/chime9_mcorec/manifests/mcorec_dev_nemo_manifest.jsonl \
    model.tokenizer.dir=${tokenizers_path}/ls960/tokenizer_spe_bpe_v500 \
    model.tokenizer.type="bpe" \
    trainer.devices=-1 \
    +evaluate_at_start=True \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=1000 \
    trainer.log_every_n_steps=10 \
    model.optim.sched.warmup_steps=2000 \
    model.optim.lr=0.05 \
    model.train_ds.batch_size=4 \
    trainer.accumulate_grad_batches=1 \
    model.validation_ds.batch_size=1 \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.exp_dir=/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/misc/nemo_experiments/debug2 \
    +init_from_ptl_ckpt=/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/misc/nemo_experiments/parakeet_v2_0.6b_nsf_test_fddt_lr_mul_100_lr0.5_dicow_data_setup_nsf_dev_longform_2k_warmup_v3/checkpoints/parakeet_v2_0.6b_nsf_test_fddt_lr_mul_100_lr0.5_dicow_data_setup_nsf_dev_longform_2k_warmup_v3--val/best.ckpt \
    exp_manager.checkpoint_callback_params.monitor="val/cp_wer" \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="pretrained_parakeet_v2_dcwsetup_ft_mcorec_sg_trimmed_new_env" \
    exp_manager.wandb_logger_kwargs.project="dk_chime9" \
    +init_from_pretrained="nvidia/parakeet-tdt-0.6b-v2" \
    name="pretrained_parakeet_v2_dcwsetup_ft_mcorec_sg_trimmed_v2_new_env" \
    +model.decoding.compute_timestamps=True \
    +model.decoding.preserve_alignments=True
