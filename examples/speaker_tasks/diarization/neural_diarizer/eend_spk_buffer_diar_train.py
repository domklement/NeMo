# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import lightning.pytorch as pl
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import seed_everything

from nemo.collections.asr.models.eend_spk_buffer_diar_model import EENDSpkBuffEncLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
Example training session (single node training)

python ./sortformer_diar_train.py --config-path='../conf/neural_diarizer' \
    --config-name='sortformer_diarizer_hybrid_loss_4spk-v1.yaml' \
    trainer.devices=1 \
    model.train_ds.manifest_filepath="<train_manifest_path>" \
    model.validation_ds.manifest_filepath="<dev_manifest_path>" \
    exp_manager.name='sample_train' \
    exp_manager.exp_dir='./sortformer_diar_train'
"""

seed_everything(42)


@hydra_runner(config_path="../conf/neural_diarizer", config_name="sortformer_diarizer_hybrid_loss_4spk-v1.yaml")
def main(cfg):
    """Main function for training the sortformer diarizer model."""
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)

    if cfg.get('load_everything_from_ptl_ckpt', False):
        from torch import load as torch_load
        x = torch_load(cfg.load_everything_from_ptl_ckpt, weights_only=False)
        model_cfg = dict(x['hyper_parameters']['cfg'])
        model_cfg.pop('train_ds')
        model_cfg.pop('validation_ds')
        model_cfg.pop('test_ds')
        model_cfg.pop('optim')
        cfg.model = {**cfg.model, **model_cfg}

    exp_manager(trainer, cfg.get("exp_manager", None))
    eend_model = EENDSpkBuffEncLabelModel(cfg=cfg.model, trainer=trainer)

    if cfg.get('init_from_nest', False) or cfg.get('init_conv_downsampling_from_nest', False):
        from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel
        nest_model = EncDecDenoiseMaskedTokenPredModel.from_pretrained(model_name="nvidia/ssl_en_nest_large_v1.0", map_location='cpu')

        if not cfg.get('init_from_nest', False):
            print('Loading NEST feature extractor state dict:', eend_model.encoder.pre_encode.load_state_dict(nest_model.encoder.pre_encode.state_dict(), strict=False))
        else:
            print('Loading NEST state dict:', eend_model.load_state_dict(nest_model.state_dict(), strict=False))

        if cfg.get('freeze_nest_parameters', False):
            eend_model_params = dict(eend_model.named_parameters())
            eend_model_param_names = set(eend_model_params.keys())
            frozen_params = []
            for n, _ in nest_model.named_parameters():
                if n in eend_model_param_names:
                    frozen_params.append(n)
                    eend_model_params[n].requires_grad = False
            print(f'Frozen {len(frozen_params)} parameters: {frozen_params}')
    
    if cfg.get('freeze_conv_downsampling', False):
        for p in eend_model.encoder.pre_encode.parameters():
            p.requires_grad = False

    if cfg.get('load_everything_from_ptl_ckpt', False):
        print('Loading everything from pretrained ckpt:', eend_model.load_state_dict(x['state_dict']))
    else:
        eend_model.maybe_init_from_pretrained_checkpoint(cfg)

    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.watch(eend_model, log="all", log_freq=500, log_graph=False)
        # trainer.logger.run.log_code(".")

    if cfg.get('evaluate_at_start', False) or cfg.get('decode_only', False):
        trainer.validate(eend_model)

    if cfg.get('decode_only', False):
        return

    # trainer.validate(eend_model)
    trainer.fit(eend_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if eend_model.prepare_test(trainer):
            trainer.test(eend_model)


if __name__ == '__main__':
    main()
