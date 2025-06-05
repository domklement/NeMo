import re


class MLCTextNormalizer:
    """

    """

    def __init__(self):
        custom_punctuations = r'!"#$%&()*+,./:;<=>?@[\\]^_`{|}~。、？！・¿¡，'
        self.punctuation_pattern = re.compile(f'[{re.escape(custom_punctuations)}]')

    def __call__(self, input_text: str):
        text = input_text.lower()
        text_tn = self.punctuation_pattern.sub('', text)
        text_tn = re.sub(r' +', ' ', text_tn)
        return text_tn
