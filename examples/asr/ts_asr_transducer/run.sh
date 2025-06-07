#!/bin/bash

export WANDB_API_KEY="029a7cc91a821854880a156d05653e9b79d9e7d6"
export WANDB_MODE="online"


python ~/NeMo/examples/asr/ts_asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path="/home/jovyan/NeMo/examples/asr/conf/fastconformer/hybrid_transducer_ctc" \
    --config-name="fastconformer_hybrid_tdt_ctc_bpe_stno.yaml" \
    model.train_ds.manifest_filepath=/tmp/nsf_nemo_train_sc_30s.jsonl \
    model.validation_ds.manifest_filepath=/tmp/nsf_nemo_train_sc_30s_small.jsonl \
    model.tokenizer.dir=/home/jovyan/NeMo/tokenizers/ls960/tokenizer_spe_bpe_v500 \
    model.tokenizer.type="bpe" \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=1200 \
    model.encoder.d_model=256 \
    model.encoder.n_heads=4 \
    model.encoder.n_layers=16 \
    model.encoder.conv_kernel_size=9 \
    model.optim.weight_decay=1e-3 \
    model.optim.lr=7e-4 \
    trainer.log_every_n_steps=20 \
    trainer.check_val_every_n_epoch=1 \
    model.train_ds.batch_size=64 \
    model.validation_ds.batch_size=32 \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.exp_dir="/home/jovyan/NeMo/nemo_experiments" \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="fastconformer_hybrid_tdt_ctc_bpe_stno_nsf_test_v2" \
    exp_manager.wandb_logger_kwargs.project="nemo_tests" \
    exp_manager.checkpoint_callback_params.monitor="val/cp_wer" \
    name="FastConformer-Hybrid-TDT-CTC-BPE-STNO-cpwer_scoring" \
    +model.decoding.compute_timestamps=True
    # +model.decoding.strategy=greedy
