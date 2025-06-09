#!/bin/bash

export WANDB_API_KEY="029a7cc91a821854880a156d05653e9b79d9e7d6"
export WANDB_MODE="online"

# manifest_path="/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/misc/manifests"
manifest_path="/tmp/dicow_data/nemo_manifests"
tokenizers_path="/tmp/tokenizers"


# python /storage/brno12-cerit/home/dklement/speech/ASR/NeMo/examples/asr/ts_asr_transducer/speech_to_text_rnnt_bpe.py \
#     --config-path="/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/examples/asr/conf/fastconformer/hybrid_transducer_ctc" \
#     --config-name="fastconformer_hybrid_tdt_ctc_bpe_stno.yaml" \
#     model.train_ds.manifest_filepath=${manifest_path}/nsf_nemo_train_sc_30s.jsonl \
#     model.validation_ds.manifest_filepath=${manifest_path}/nsf_nemo_train_sc_30s_small.jsonl \
#     model.tokenizer.dir=${tokenizers_path}/ls960/tokenizer_spe_bpe_v500 \
#     model.tokenizer.type="bpe" \
#     trainer.devices=4 \
#     trainer.accelerator="gpu" \
#     trainer.strategy="ddp" \
#     trainer.max_epochs=1000 \
#     model.encoder.d_model=256 \
#     model.encoder.n_heads=8 \
#     model.encoder.n_layers=16 \
#     trainer.log_every_n_steps=20 \
#     trainer.check_val_every_n_epoch=5 \
#     model.train_ds.batch_size=32 \
#     model.validation_ds.batch_size=64 \
#     exp_manager.resume_if_exists=True \
#     exp_manager.resume_ignore_no_checkpoint=True \
#     exp_manager.exp_dir="/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/misc/nemo_experiments" \
#     exp_manager.checkpoint_callback_params.monitor="val/cp_wer" \
#     name="FastConformer-Hybrid-TDT-CTC-BPE-STNO-cpwer_scoring_dgx_4gpu_v5" \
#     +model.decoding.compute_timestamps=True \
#     +model.decoding.preserve_alignments=True \
#     exp_manager.create_wandb_logger=True \
#     exp_manager.wandb_logger_kwargs.name="fastconformer_hybrid_tdt_ctc_bpe_stno_nsf_test_dgx_4gpu_v5" \
#     exp_manager.wandb_logger_kwargs.project="nemo_tests" \
#     # model.encoder.conv_kernel_size=9 \
#     # model.optim.weight_decay=1e-3 \
#     # model.optim.lr=1e-3 \
#     # +model.decoding.strategy=greedy

# python /storage/brno12-cerit/home/dklement/speech/ASR/NeMo/examples/asr/ts_asr_transducer/speech_to_text_rnnt_bpe.py \
#     --config-path="/storage/brno12-cerit/home/dklement/speech/ASR/NeMo/examples/asr/conf/fastconformer/hybrid_transducer_ctc" \
#     --config-name="fastconformer_hybrid_tdt_ctc_bpe_stno_parakeet_v2_0.6b.yaml" \
#     model.train_ds.manifest_filepath=${manifest_path}/nsf_nemo_train_sc_30s.jsonl \
#     model.validation_ds.manifest_filepath=${manifest_path}/nsf_nemo_train_sc_30s_small.jsonl \
#     model.tokenizer.dir=${tokenizers_path}/ls960/tokenizer_spe_bpe_v500 \
#     model.tokenizer.type="bpe" \
#     trainer.devices=-1 \
#     trainer.accelerator="gpu" \
#     trainer.strategy="ddp" \
#     trainer.max_epochs=1000 \
#     trainer.log_every_n_steps=20 \
#     trainer.check_val_every_n_epoch=1 \
#     model.train_ds.batch_size=16 \
#     trainer.accumulate_grad_batches=4 \
#     model.validation_ds.batch_size=1 \
#     exp_manager.resume_if_exists=True \
#     exp_manager.resume_ignore_no_checkpoint=True \
#     exp_manager.exp_dir="/tmp/nemo_experiments" \
#     exp_manager.checkpoint_callback_params.monitor="val/cp_wer" \
#     exp_manager.create_wandb_logger=True \
#     exp_manager.wandb_logger_kwargs.name="parakeet_v2_0.6b_nsf_test_fddt_lr_mul_100" \
#     exp_manager.wandb_logger_kwargs.project="nemo_tests" \
#     +init_from_pretrained="nvidia/parakeet-tdt-0.6b-v2" \
#     name="parakeet_v2_0.6b_nsf_test_fddt_lr_mul_100" \
#     +model.decoding.compute_timestamps=True \
#     +model.decoding.preserve_alignments=True \
    # model.encoder.conv_kernel_size=9 \
    # model.optim.weight_decay=1e-3 \
    # model.optim.lr=1e-3 \
    # +model.decoding.strategy=greedy

python /home/jovyan/NeMo/examples/asr/ts_asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path="/home/jovyan/NeMo/examples/asr/conf/fastconformer/hybrid_transducer_ctc" \
    --config-name="fastconformer_hybrid_tdt_ctc_bpe_stno_parakeet_v2_0.6b.yaml" \
    model.train_ds.manifest_filepath=${manifest_path}/ami_nsf_l2m_both_100_360_compound_trainset.jsonl \
    model.validation_ds.manifest_filepath=${manifest_path}/notsofar_eval_sc_cutset.jsonl \
    model.tokenizer.dir=${tokenizers_path}/ls960/tokenizer_spe_bpe_v500 \
    model.tokenizer.type="bpe" \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=1000 \
    trainer.log_every_n_steps=20 \
    trainer.val_check_interval=0.50 \
    model.train_ds.batch_size=16 \
    trainer.accumulate_grad_batches=4 \
    model.validation_ds.batch_size=1 \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.exp_dir="/tmp/nemo_experiments" \
    exp_manager.checkpoint_callback_params.monitor="val/cp_wer" \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="parakeet_v2_0.6b_nsf_test_fddt_lr_mul_10_dicow_data_setup_nsf_dev_longform_v3" \
    exp_manager.wandb_logger_kwargs.project="nemo_tests" \
    +init_from_pretrained="nvidia/parakeet-tdt-0.6b-v2" \
    name="parakeet_v2_0.6b_nsf_test_fddt_lr_mul_10_dicow_data_setup_nsf_dev_longform_v3" \
    +model.decoding.compute_timestamps=True \
    +model.decoding.preserve_alignments=True