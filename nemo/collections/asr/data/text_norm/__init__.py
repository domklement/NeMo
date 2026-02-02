"""
NOTSOFAR adopts the same text normalizer as the CHiME-8 DASR track.
This code is aligned with the CHiME-8 repo:
https://github.com/chimechallenge/chime-utils/tree/main/chime_utils/text_norm
"""
import json
import os
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from nemo_text_processing.text_normalization.normalize import Normalizer
from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizerNSF
from .mlc_norm import MLCTextNormalizer
from .remove_disfluencies import remove_disfluencies


def get_text_norm(t_norm: str):
    if t_norm == 'whisper_basic':
        return EnglishTextNormalizer({})
    elif t_norm == 'whisper_basic_rm_disf':
        normalizer = EnglishTextNormalizer({})
        def whisper_basic_rm_disf(text):
            return remove_disfluencies(normalizer(text))
        return whisper_basic_rm_disf
    elif t_norm == 'whisper':
        SPELLING_CORRECTIONS = json.load(open(f'{os.path.dirname(__file__)}/english.json'))
        return EnglishTextNormalizer(SPELLING_CORRECTIONS)
    elif t_norm == 'whisper_nsf':
        return EnglishTextNormalizerNSF()
    elif t_norm == 'mlc-slm':
        return MLCTextNormalizer()
    elif t_norm == 'nemo_en_cased':
        normalizer = Normalizer(input_case='cased', lang='en')
        return normalizer.normalize
    else:
        raise ValueError(f"Unsupported text normalization type: {t_norm}")
