# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.models import EncDecRNNTBPEModelSTNOAV
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.collections.asr.models import ASRModel

from pytorch_lightning.callbacks import Callback
class EvalAtStartCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Evaluating at start...")
        trainer.validate(pl_module)

@hydra_runner(config_path="../conf/fastconformer/hybrid_transducer_ctc", config_name="fastconformer_hybrid_tdt_ctc_bpe_stno")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    # trainer.callbacks.append(EvalAtStartCallback())
    init_from_pretrained = cfg.get("init_from_pretrained", None)
    pretrained_model = None
    if init_from_pretrained is not None:
        pretrained_model = ASRModel.from_pretrained(model_name=init_from_pretrained, map_location='cpu')

    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecRNNTBPEModelSTNOAV(cfg=cfg.model, trainer=trainer, tokenizer=pretrained_model.tokenizer if pretrained_model is not None else None)

    if init_from_pretrained is not None:
        missing, unexpected = asr_model.load_state_dict(pretrained_model.state_dict(), strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    # asr_model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=(256, 256))

    if cfg.get("decode_only", False):
        trainer.validate(asr_model)
        return

    if cfg.get("evaluate_at_start", True):
        trainer.validate(asr_model)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
