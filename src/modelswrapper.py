

from sw_core import sw_logger
import sw_constants
import gensim
import os
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import nltk
from allennlp.modules.elmo import Elmo, batch_to_ids

# TODO: переделать на классы, в которых и модели, и параметры, здесь просто создавать экземпляры

from abc import ABCMeta, abstractmethod


# Чтобы ко всем моделям можно было обращаться одним образом, опишем интерфейс модели
def abstractfunc(func):
    func.__isabstract__ = True
    return func


class IAbstractModelWrapper(type):

    def __init__(self, name, bases, namespace):
        for base in bases:
            must_implement = getattr(base, 'abstract_methods', [])
            class_methods = getattr(self, 'all_methods', [])
            for method in must_implement:
                if method not in class_methods:
                    err_str = """Can't create abstract class {name}!
                    {name} must implement abstract method {method} of class {base_class}!""".format(name=name,
                                                                                                    method=method,
                                                                                                    base_class=base.__name__)
                    raise TypeError(err_str)

    def __new__(metaclass, name, bases, namespace):
        namespace['abstract_methods'] = IAbstractModelWrapper._get_abstract_methods(namespace)
        namespace['all_methods'] = IAbstractModelWrapper._get_all_methods(namespace)
        cls = super().__new__(metaclass, name, bases, namespace)
        return cls

    def _get_abstract_methods(namespace):
        return [name for name, val in namespace.items() if callable(val) and getattr(val, '__isabstract__', False)]

    def _get_all_methods(namespace):
        return [name for name, val in namespace.items() if callable(val)]


class BaseModelWrapper(metaclass=IAbstractModelWrapper):

    @abstractfunc
    def __init__(self, model_name):
        self.model = None
        self.tokenizer = None
        if model_name in sw_constants.SW_SUPPORTED_MODELS:
            if (model_name in sw_constants.SW_BERT_MODELS) or (model_name in sw_constants.SW_WORD2VEC_MODELS):
                self.model_name = model_name
                self.__model_name_or_path__ = sw_constants.SW_SUPPORTED_MODELS[model_name]
        else:
            raise Exception('Unsupported model type!')

    @abstractfunc
    def get_embeddings(self, word):
        pass

    @abstractfunc
    def check_init_model_state(self):
        if self.model is None:
            if self.model_name in sw_constants.SW_WORD2VEC_MODELS:
               # sw_logger.info(
               #     'Нам потребовалась модель, которая не была загружена:' + self.model_name + 'Загружаем...')
                self.model = gensim.models.KeyedVectors.load(sw_constants.SW_WORD2VEC_MODELS[self.model_name])
               # sw_logger.info('Загрузка завершена.')

            if self.model_name in sw_constants.SW_BERT_MODELS:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    sw_constants.SW_SUPPORTED_MODELS[self.model_name])
                self.model = \
                    AutoModelWithLMHead.from_pretrained(sw_constants.SW_SUPPORTED_MODELS[self.model_name])

class BertModelWrapper(BaseModelWrapper):

    def __init__(self, model):
        super(BertModelWrapper, self).__init__(model)

    def get_embeddings(self, word):
        super(self.__class__, self).check_init_model_state()
        input_ids = self.tokenizer.encode(word, return_tensors="pt")
        out = self.model(input_ids)
        embeddings = out[1][1][:, -1, :].detach().numpy().tolist()
        return embeddings

    def check_init_model_state(self):
        pass




sw_logger.info('Начинаем загрузку моделей...')

sw_logger.info('Загружаем модель CONVERSATIONAL_RU_BERT')
conversational_ru_bert_tokenizer = AutoTokenizer.from_pretrained(sw_constants.CONVERSATIONAL_RU_BERT_MODEL_PATH)
conversational_ru_bert_model = \
    AutoModelWithLMHead.from_pretrained(sw_constants.CONVERSATIONAL_RU_BERT_MODEL_PATH)

sw_logger.info('Загружаем модель RU_BERT_CASED_MODEL')
ru_bert_cased_tokenizer = AutoTokenizer.from_pretrained(sw_constants.RU_BERT_CASED_MODEL_PATH)
ru_bert_cased_model = \
    AutoModelWithLMHead.from_pretrained(sw_constants.RU_BERT_CASED_MODEL_PATH)

sw_logger.info('Загружаем модель SENTENCE_RU_BERT')
sentence_ru_bert_tokenizer = AutoTokenizer.from_pretrained(sw_constants.SENTENCE_RU_BERT_MODEL_PATH)
sentence_ru_bert_model = \
    AutoModelWithLMHead.from_pretrained(sw_constants.SENTENCE_RU_BERT_MODEL_PATH)

sw_logger.info('Загружаем модель SLAVIC_BERT')
slavic_bert_tokenizer = AutoTokenizer.from_pretrained(sw_constants.SLAVIC_BERT_MODEL_PATH)
slavic_bert_model = \
    AutoModelWithLMHead.from_pretrained(sw_constants.SLAVIC_BERT_MODEL_PATH)

sw_logger.info('Загружаем модель BERT_BASE_MULTILINGUAL_UNCASED')
bert_base_multilingual_uncased_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
bert_base_multilingual_uncased_model = AutoModelWithLMHead.from_pretrained('bert-base-multilingual-uncased')

sw_logger.info('Загружаем модель BERT_BASE_MULTILINGUAL_CASED')
bert_base_multilingual_cased_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_base_multilingual_cased_model = AutoModelWithLMHead.from_pretrained('bert-base-multilingual-cased')

sw_logger.info('Загружаем модель ELMo: tayga_lemmas_elmo_2048_2019')
elmo_model = Elmo(sw_constants.ELMO_MODEL_OPTIONS_FILE, sw_constants.ELMO_MODEL_WEIGHTS_FILE, 2, dropout=0)

sw_logger.info('Загружаем модель GPT: Russian GPT2 finetuning')
gpt2_tokenizer = AutoTokenizer.from_pretrained(sw_constants.GPT2_MODEL_PATH)
gpt2_model = AutoModelWithLMHead.from_pretrained(sw_constants.GPT2_MODEL_PATH)

sw_logger.info('Загружаем модель Word2Vec (tayga, fasttext, B-o-W)')
word2vec_model_file = os.path.join(sw_constants.WORD2VEC_MODEL_PATH, sw_constants.WORD2VEC_MODEL_FILE)
word2vec_wrapper = gensim.models.KeyedVectors.load(word2vec_model_file)

sw_logger.info('Все модели загружены успешно!')

# sw_logger.info('Загружаем пакет «stopwords» для nltk')
# nltk.download("stopwords")

# TODO: Предсказатель и эмбеддинги на GPT-2
# TODO: Эмбеддинги на ELMO без Deeppavlov скрипт конвертации: https://github.com/vlarine/transformers-ru

# TODO: Добавить ELMo и GPT2

# TODO: списки / словари моделей и токенайзеров, интерфейсы MODEL и TOKENIZER

# TODO: Загрузка GPT2 и ELMO, подумать над загрузкой on demand (сунуть загрузку в метод, выполняемый при выполнении
# любого метода класса)

# TODO: most_similar_to_given - посмотреть, как сделано, реализовать (либо так же, либо через векторную СУБД,
#  либо через какую-то быструю структуру поиска по векторам — на словаре Тайги и на нашем словаре)

# TODO: Посмотреть на most_similar_to_given как на задачу классификации, попробовать k_p_чего-то там для поиска
#  наиболее верояных слов из набора (а не вообще)
#
