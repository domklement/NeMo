#!/bin/bash

export WANDB_API_KEY="029a7cc91a821854880a156d05653e9b79d9e7d6"
export WANDB_MODE="online"

python ~/NeMo/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
    --config-path="/home/jovyan/NeMo/examples/asr/conf/fastconformer" \
    --config-name="fast-conformer_ctc_bpe.yaml" \
    model.train_ds.manifest_filepath=/tmp/librispeech/train_clean_5.json \
    model.validation_ds.manifest_filepath=/tmp/librispeech/dev_clean_2.json \
    model.tokenizer.dir=/home/jovyan/NeMo/tokenizers/ls960/tokenizer_spe_bpe_v500 \
    model.tokenizer.type="bpe" \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=100 \
    model.encoder.d_model=256 \
    model.encoder.n_heads=4 \
    model.encoder.n_layers=16 \
    model.encoder.conv_kernel_size=9 \
    model.optim.weight_decay=1e-3 \
    +model.decoding.compute_timestamps=True \
    +model.decoding.preserve_alignments=True \
    trainer.val_check_interval=500 \
    trainer.log_every_n_steps=20 \
    model.train_ds.batch_size=256 \
    model.validation_ds.batch_size=32 \
    exp_manager.create_wandb_logger=False \
    exp_manager.create_tensorboard_logger=False

# python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
#     --config-path="/home/jovyan/NeMo/examples/asr/conf/fastconformer" \
#     --config-name="fast-conformer_ctc_bpe.yaml" \
#     model.train_ds.manifest_filepath=/tmp/librispeech/lhotse_manifests/librispeech_cuts_train-other-500.jsonl.gz \
#     model.validation_ds.manifest_filepath=/tmp/librispeech/lhotse_manifests/librispeech_cuts_dev-other.jsonl.gz \
#     model.tokenizer.dir=/tmp/librispeech/tokenizer_spe_bpe_v500 \
#     model.tokenizer.type="bpe" \
#     trainer.devices=-1 \
#     trainer.accelerator="gpu" \
#     trainer.strategy="ddp" \
#     trainer.max_epochs=100 \
#     model.encoder.d_model=176 \
#     model.encoder.n_heads=8 \
#     model.encoder.n_layers=16 \
#     model.encoder.conv_kernel_size=9 \
#     model.optim.weight_decay=0.0 \
#     model.train_ds.batch_size=256 \
#     model.validation_ds.batch_size=32 \
#     exp_manager.create_wandb_logger=True \
#     ++model.train_ds.use_lhotse=True \
#     ++model.validation_ds.use_lhotse=True \
#     ++model.train_ds.batch_duration=1100 \
#     ++model.train_ds.quadratic_duration=30 \
#     ++model.train_ds.num_buckets=30 \
#     ++model.train_ds.num_cuts_for_bins_estimate=10000 \
#     ++model.train_ds.bucket_buffer_size=10000 \
#     ++model.train_ds.shuffle_buffer_size=10000 \
#     ++trainer.use_distributed_sampler=false \
#     ++trainer.limit_train_batches=1000 \
#     exp_manager.wandb_logger_kwargs.name="fastconformer_ctc_500bpe_ls960_small_honza_256bs_v2_lhotse" \
#     exp_manager.wandb_logger_kwargs.project="nemo_tests" \
