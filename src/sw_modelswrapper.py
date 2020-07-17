from sw_core import sw_logger
import sw_constants
import gensim
import os, pathlib, json
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, BertModel
import nltk
from allennlp.modules.elmo import Elmo, batch_to_ids


class SWConfigParser:
    def __init__(self, config_file_name=sw_constants.SW_CONFIG_FILE_NAME):
        __project_path__ = pathlib.Path(__file__).parent.absolute()
        config_file = os.path.join(__project_path__, config_file_name)
        with open(config_file, 'r') as fp:
            data = json.load(fp)
        fp.close()
        sw_supported_models = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    if isinstance(v, list):
                        i = 0
                        for element in v:
                            if os.path.isdir(str('.' + element)) or os.path.isfile(str('.' + element)):
                                v[i] = os.path.join(__project_path__, str('.' + v[i])).replace('/./', '/')
                                i = i + 1
                        sw_supported_models = {**sw_supported_models, **value}
                    elif os.path.isdir(str('.' + v)) or os.path.isfile(str('.' + v)):
                        value[n] = os.path.join(__project_path__, str('.' + v)).replace('/./', '/')
                sw_supported_models = {**sw_supported_models, **value}



        all_models = {'sw_supported_models': sw_supported_models}
        data = {**data, **all_models}
        self.config = data.copy()


config_parser = SWConfigParser()

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
        if model_name in config_parser.config['sw_supported_models']:
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
            if self.model_name in config_parser.config['sw_word2vec_models']:
                self.model = gensim.models.KeyedVectors.load(config_parser.config['sw_word2vec_models'][self.model_name])

            if (self.model_name in config_parser.config['sw_bert_models']) or (self.model_name in config_parser.config['sw_gpt2_models']):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config_parser.config['sw_supported_models'][self.model_name])
                self.model = \
                    AutoModelWithLMHead.from_pretrained(config_parser.config['sw_supported_models'][self.model_name])

            if self.model_name in config_parser.config['sw_elmo_models']:
                self.model = Elmo(config_parser.config['sw_supported_models'][self.model_name][0],
                                  config_parser.config['sw_supported_models'][self.model_name][1], 2,
                                  dropout=0)

            if self.model_name in config_parser.config['sw_gpt2_models']:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config_parser.config['sw_supported_models'][self.model_name])
                self.model = \
                    AutoModelWithLMHead.from_pretrained(config_parser.config['sw_supported_models'][self.model_name])

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


class gpt2ModelWrapper(BaseModelWrapper):
    def __init__(self, model):
        super(self.__class__, self).__init__(model)

    def get_embeddings(self, word):
        super(self.__class__, self).check_init_model_state()
        inputs = self.tokenizer.encode(word, return_tensors="pt")
        outputs = self.model.transformer.wte.weight[inputs, :][0][0].detach().numpy().reshape(1, -1).tolist()[0]
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


all_sw_models = {}
for key, value in config_parser.config['sw_supported_models'].items():
    if 'bert' in key:
        all_sw_models[key] = BertModelWrapper(key)
    if 'elmo' in key:
        all_sw_models[key] = ELMoModelWrapper(key)
    if 'word2vec' in key:
        all_sw_models[key] = Word2VecModelWrapper(key)
    if 'gpt2' in key:
        all_sw_models[key] = gpt2ModelWrapper(key)


# # Экземпляры моделей для использования в остальных модклях. Готовые. Реальная загрузка — по требованию.
# word2vec_tayga_bow_model = Word2VecModelWrapper(config_parser.config['sw_word2vec_models']['word2vec_TAYGA_BOW_NAME'])
# conversational_ru_bert_model = BertModelWrapper(list(config_parser.config['sw_bert_models'].keys())[2])
# ru_bert_cased_model = BertModelWrapper(config_parser.config['sw_bert_models']['sentence_ru_bert_NAME'])
# sentence_ru_bert_model = BertModelWrapper(config_parser.config['sw_bert_models']['sentence_ru_bert_NAME'])
# slavic_bert_model_model = BertModelWrapper(config_parser.config['sw_bert_models']['SLAVIC_bert_NAME'])
# bert_base_multilingual_uncased_model = BertModelWrapper(list(config_parser.config['sw_bert_models'].keys())[1])
# bert_base_multilingual_cased_model = BertModelWrapper(list(config_parser.config['sw_bert_models'].keys())[0])
# elmo_tayga_lemmas_2048_model = BertModelWrapper(config_parser.config['sw_elmo_models']['elmo_TAYGA_LEMMAS_2048_NAME'])
# gpt2_russian_finetuning_model = BertModelWrapper(config_parser.config['sw_gpt2_models']['gpt2_RUSSIAN_FINETUNING_NAME'])

# TODO: Предсказатель и эмбеддинги на GPT-2
# TODO: Эмбеддинги на elmo без Deeppavlov скрипт конвертации: https://github.com/vlarine/transformers-ru

# TODO: Добавить ELMo и gpt2

# TODO: списки / словари моделей и токенайзеров, интерфейсы MODEL и TOKENIZER

# TODO: Загрузка gpt2 и elmo, подумать над загрузкой on demand (сунуть загрузку в метод, выполняемый при выполнении
# любого метода класса)

# TODO: most_similar_to_given - посмотреть, как сделано, реализовать (либо так же, либо через векторную СУБД,
#  либо через какую-то быструю структуру поиска по векторам — на словаре Тайги и на нашем словаре)

# TODO: Посмотреть на most_similar_to_given как на задачу классификации, попробовать k_p_чего-то там для поиска
#  наиболее верояных слов из набора (а не вообще)
#
