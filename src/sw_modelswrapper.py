from sw_core import sw_logger
import sw_constants
import gensim
import os
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, BertModel
import nltk
from allennlp.modules.elmo import Elmo, batch_to_ids

# TODO: переделать на классы, в которых и модели, и параметры, здесь просто создавать экземпляры

# Чтобы ко всем моделям можно было обращаться одним образом, опишем интерфейс модели
def abstractfunc(func):
    func.__isabstract__ = True
    return func


# TODO разрешить далёким потомкам не иметь метода check_init_model_state
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
            self.model_name = model_name
        else:
            raise Exception('Unsupported model type!')

    @abstractfunc
    def get_embeddings(self, word):
        pass

    @abstractfunc
    def check_init_model_state(self):
        if self.model is None:
            sw_logger.info(
                'Нам потребовалась модель, которая не была загружена:' +
                ' {model_name}, загружаем её...'.format(model_name=self.model_name))
            if self.model_name in sw_constants.SW_WORD2VEC_MODELS:
                self.model = gensim.models.KeyedVectors.load(sw_constants.SW_WORD2VEC_MODELS[self.model_name])

            if (self.model_name in sw_constants.SW_BERT_MODELS) or (self.model_name in sw_constants.SW_GPT2_MODELS):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    sw_constants.SW_SUPPORTED_MODELS[self.model_name])
                self.model = \
                    AutoModelWithLMHead.from_pretrained(sw_constants.SW_SUPPORTED_MODELS[self.model_name])

            if self.model_name in sw_constants.SW_ELMO_MODELS:
                self.model = Elmo(sw_constants.SW_SUPPORTED_MODELS[self.model_name][0],
                                  sw_constants.SW_SUPPORTED_MODELS[self.model_name][1], 2,
                                  dropout=0)

            if self.model_name in sw_constants.SW_GPT2_MODELS:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    sw_constants.SW_SUPPORTED_MODELS[self.model_name])
                self.model = \
                    AutoModelWithLMHead.from_pretrained(sw_constants.SW_SUPPORTED_MODELS[self.model_name])


            sw_logger.info('Загрузка модели {model_name} завершена.'.format(model_name=self.model_name))


class BertModelWrapper(BaseModelWrapper):

    def __init__(self, model):
        super(self.__class__, self).__init__(model)

    def get_embeddings(self, word):
        super(self.__class__, self).check_init_model_state()
        input_ids = self.tokenizer.encode(word, return_tensors="pt")
        out = self.model(input_ids)
        embeddings = out[1][1][:, -1, :].detach().numpy().tolist()
        return embeddings

    def check_init_model_state(self):
        pass


class Word2VecModelWrapper(BaseModelWrapper):
    def __init__(self, model):
        super(self.__class__, self).__init__(model)

    def get_embeddings(self, word):
        super(self.__class__, self).check_init_model_state()
        return self.model.get_vector(word)
        pass

    def check_init_model_state(self):
        pass


class GPT2ModelWrapper(BaseModelWrapper):
    def __init__(self, model):
        super(self.__class__, self).__init__(model)

    def get_embeddings(self, word):
        super(self.__class__, self).check_init_model_state()
        inputs = self.tokenizer.encode(word, return_tensors="pt")
        outputs = self.model.transformer.wte.weight[inputs,:][0][0].detach().numpy().reshape(1, -1).tolist()[0]
        return outputs

    def check_init_model_state(self):
        pass


class ELMoModelWrapper(BaseModelWrapper):
    def __init__(self, model):
        super(self.__class__, self).__init__(model)

    def get_embeddings(self, word):
        super(self.__class__, self).check_init_model_state()
        character_ids = batch_to_ids(word)
        embeddings = self.model(character_ids)
        return embeddings['elmo_representations'][1][0][0].detach().numpy().reshape(1, -1).tolist()


    def check_init_model_state(self):
        pass


"""word2vec_tayga_bow_model = Word2VecModelWrapper(sw_constants.WORD2VEC_TAYGA_BOW_NAME)
conversational_ru_bert_model = BertModelWrapper(sw_constants.CONVERSATIONAL_RU_BERT_NAME)
RU_BERT_CASED_NAME

RU_BERT_CASED_NAME = 'RU_BERT_CASED'
SENTENCE_RU_BERT_NAME = 'SENTENCE_RU_BERT'
SLAVIC_BERT_MODEL_NAME = 'SLAVIC_BERT_MODEL'
BERT_BASE_MULTILINGUAL_UNCASED_NAME = 'BERT_BASE_MULTILINGUAL_UNCASED'
BERT_BASE_MULTILINGUAL_CASED_NAME = 'BERT_BASE_MULTILINGUAL_CASED'
ELMO_TAYGA_LEMMAS_2048_NAME = 'ELMO_TAYGA_LEMMAS_2048'
GPT2_RUSSIAN_FINETUNING_NAME = """

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
