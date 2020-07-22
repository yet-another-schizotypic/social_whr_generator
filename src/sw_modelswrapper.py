import math
from random import randint
import re
from sw_core import sw_logger
from sw_core import config_parser
import itertools
import gensim
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, BertModel

import nltk
from allennlp.modules.elmo import Elmo, batch_to_ids
import json, os
import scipy
from Levenshtein import distance as levenshtein_distance
from sw_core import Math


class Word(object):
    def __init__(self, title):
        self.__title__ = title.lower()
        self.__embeddings__ = None
        self.__suggested_by__ = None
        self.__tested_as_target__ = None
        self.__tested_in_explanation_chains__ = None
        self.__succeeded_as_target__ = None
        self.__succeeded_in_explanation_chains__ = None

    def get_word_embeddings(self, model_name):
        if self.__embeddings__ is None:
            self.__embeddings__ = {}
            for sw_model_name in config_parser.config['sw_supported_models'].keys():
                if model_name == sw_model_name:
                    self.__embeddings__[model_name] = all_sw_models[model_name].get_embeddings(self.__title__)
                else:
                    self.__embeddings__[sw_model_name] = None
        elif self.__embeddings__[model_name] is None:
            self.__embeddings__[model_name] = all_sw_models[model_name].get_embeddings(self.__title__)

        return self.__embeddings__[model_name]

    @property
    def title(self):
        return self.__title__

    @property
    def suggested_by(self):
        return self.__suggested_by__

    @suggested_by.setter
    def suggested_by(self, value):
        self.__suggested_by__ = value

    @property
    def tested_as_target(self):
        return self.__tested_as_target__

    @tested_as_target.setter
    def tested_as_target(self, value):
        self.__tested_as_target__ = value

    @property
    def tested_in_explanation_chains(self):
        return self.__tested_in_explanation_chains__

    @tested_in_explanation_chains.setter
    def tested_in_explanation_chains(self, value):
        self.__tested_in_explanation_chains__ = value

    @property
    def succeeded_as_target(self):
        return self.__succeeded_as_target__

    @succeeded_as_target.setter
    def succeeded_as_target(self, value):
        self.__succeeded_as_target__ = value

    @property
    def succeeded_in_explanation_chains(self):
        return self.__succeeded_in_explanation_chains__

    @succeeded_in_explanation_chains.setter
    def succeeded_in_explanation_chains(self, value):
        self.__succeeded_in_explanation_chains__ = value


    #TODO: геттеры и сеттеры для остальных

class NLWrapper:
    def __init__(self):
        self.__frequency_dicts__ = {}
        for key, value in config_parser.config['sw_frequency_dicts'].items():
            with open(value['path'], 'r') as fp:
                data = json.load(fp)
            fp.close()
            self.__frequency_dicts__[key] = {}
            self.__frequency_dicts__[key]['data'] = data
            self.__frequency_dicts__[key]['threshold'] = value['threshold']

    def get_word_frequencies_data(self, w: Word):
        freqs = []
        for i in range(0, len(self.__frequency_dicts__)):
            dict_name = list(self.__frequency_dicts__)[i]
            w_freq = self.__frequency_dicts__[dict_name]['data'].get(w.title, -1)
            freqs.append(float(w_freq))
        return freqs

    # TODO: если что, сделать её рекурсивной для вложенных списков
    def unpack_word_objects_list(self, w_list: list):
        res = []
        for element in w_list:
            res.append(element.title)
        return res

    def unpack_word_objects_target_exp(self, target, w_list: list):
        res = [target.title]
        res.append(self.unpack_word_objects_list(w_list))
        return res

    def choose_less_frequent_word(self, w1: Word, w2: Word):
        w1_freqs = self.get_word_frequencies_data(w1)
        w2_freqs = self.get_word_frequencies_data(w2)
        for i in range(0, len(w1_freqs)):
            if w1_freqs[i] < w2_freqs[i]:
                return w1
            elif w2_freqs[i] < w1_freqs[i]:
                return w2

        for model_name in config_parser.config['sw_supported_models'].items():
            if 'word2vec' in model_name[0]:
                w1_freq = all_sw_models[model_name[0]].get_word_frequency(w1)
                w2_freq = all_sw_models[model_name[0]].get_word_frequency(w2)
                if w1_freq < w2_freq:
                    return w1
                elif w2_freq < w1_freq:
                    return w2

        if bool(randint(0, 1)):
            return w1
        else:
            return w2


nl_wrapper = NLWrapper()


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
        self.params = None
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
                if 'fasttext' in self.model_name:
                    self.model = gensim.models.fasttext.FastTextKeyedVectors.load(
                        config_parser.config['sw_word2vec_models'][self.model_name])
                else:
                    self.model = gensim.models.KeyedVectors.load(
                        config_parser.config['sw_word2vec_models'][self.model_name])
                params_file_dir = os.path.dirname(config_parser.config['sw_word2vec_models'][self.model_name])

            if (self.model_name in config_parser.config['sw_bert_models']) or (
                    self.model_name in config_parser.config['sw_gpt2_models']):
                self.tokenizer = BertTokenizer.from_pretrained(
                    config_parser.config['sw_supported_models'][self.model_name])
                self.model = \
                    BertForMaskedLM.from_pretrained(config_parser.config['sw_supported_models'][self.model_name])
                params_file_dir = config_parser.config['sw_supported_models'][self.model_name]
                self.model.eval()

            if self.model_name in config_parser.config['sw_elmo_models']:
                self.model = Elmo(config_parser.config['sw_supported_models'][self.model_name][0],
                                  config_parser.config['sw_supported_models'][self.model_name][1], 2,
                                  dropout=0)
                params_file_dir = os.path.dirname(config_parser.config['sw_supported_models'][self.model_name][0])

            if self.model_name in config_parser.config['sw_gpt2_models']:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config_parser.config['sw_supported_models'][self.model_name])
                self.model = \
                    AutoModelWithLMHead.from_pretrained(config_parser.config['sw_supported_models'][self.model_name])
                params_file_dir = config_parser.config['sw_supported_models'][self.model_name]

            params_file = os.path.join(params_file_dir, config_parser.config['sw_models_params_file'])
            with open(params_file, 'r') as fp:
                data = json.load(fp)
            fp.close()
            self.params = data.copy()
            sw_logger.info('Загрузка модели {model_name} завершена.'.format(model_name=self.model_name))

    @abstractfunc
    def check_synonymy(self, w1: Word, w2: Word):
        pass


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
        super(self.__class__, self).check_init_model_state()

    def check_synonymy(self, w1: Word, w2: Word):
        pass

    def check_explanation_chain_validity(self, target: Word, exp_chain: list):
        super(self.__class__, self).check_init_model_state()
        string = nl_wrapper.unpack_word_objects_target_exp(target, exp_chain)
        string = re.sub(r"[^а-яА-Я]+", ' ', str(string))
        tokenize_input = self.tokenizer.tokenize(string)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        predictions = self.model(tensor_input)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
        score = math.exp(loss / len(tokenize_input))

        if self.params['exp_loss_similarity_for_chain_validation_min'] <= score <= self.params['exp_loss_similarity_for_chain_validation_max'] :
            print(score)
            return True
        else:
            return False

    def check_explanation_chain_validity_with_permutations(self, target: Word, exp_chain: list):
        perms = itertools.permutations(exp_chain, len(exp_chain))
        i = 0
        for perm in perms:
            i = i + 1
            res = self.check_explanation_chain_validity(target, perm)
            if res is True:
                return True

        return False

class Word2VecModelWrapper(BaseModelWrapper):
    def __init__(self, model):
        super(self.__class__, self).__init__(model)

    def get_embeddings(self, word):
        super(self.__class__, self).check_init_model_state()
        return self.model.get_vector(word)
        pass

    def check_init_model_state(self):
        super(self.__class__, self).check_init_model_state()

    def check_synonymy(self, w1: Word, w2: Word):
        dist = scipy.spatial.distance.cosine(w1.get_word_embeddings(self.model_name),
                                             w2.get_word_embeddings(self.model_name))
        if dist < self.params['synonymy_cosine_threshold']:
            res = True
        else:
            res = False
        return res, dist

    def check_synonymy_by_relative_cosine_similarity(self, w1: Word, w2: Word):
        super(self.__class__, self).check_init_model_state()
        sim = self.model.relative_cosine_similarity(w1.title, w2.title)
        if sim > self.params['relative_cosine_similarity_threshold']:
            res = True
        else:
            res = False
        return res, sim

    def get_word_frequency(self, w: Word):
        super(self.__class__, self).check_init_model_state()
        freq = self.model.vocab.get(w.title, -1)
        if freq == -1:
            return -1
        else:
            return freq.count

    def check_explanation_chain_validity(self, target: Word, exp_chain: list):
        super(self.__class__, self).check_init_model_state()
        target_title = target.title
        positive = []
        for element in exp_chain:
            positive.append(element.title)

        suggestions = self.model.most_similar_cosmul(positive=positive)
        if target_title in suggestions:
            return True
        else:
            for s_word, s_sim in suggestions:
                sim = self.model.similarity(target_title, s_word)
                #print(target_title, s_word, sim)
                if sim >= self.params['cosmul_similarity_for_chain_validation']:
                    return True

        res_vector = []
        for element in exp_chain:
            embeddings = element.get_word_embeddings(self.model_name)
            res_vector = Math.sum_vectors(res_vector, embeddings)
            suggestions = self.model.similar_by_vector(res_vector)
            if target_title in suggestions:
                return True

        return False

class gpt2ModelWrapper(BaseModelWrapper):
    def __init__(self, model):
        super(self.__class__, self).__init__(model)

    def get_embeddings(self, word):
        super(self.__class__, self).check_init_model_state()
        inputs = self.tokenizer.encode(word, return_tensors="pt")
        outputs = self.model.transformer.wte.weight[inputs, :][0][0].detach().numpy().reshape(1, -1).tolist()[0]
        return outputs

    def check_init_model_state(self):
        super(self.__class__, self).check_init_model_state()

    def check_synonymy(self, w1: Word, w2: Word):
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
        super(self.__class__, self).check_init_model_state()

    def check_synonymy(self, w1: Word, w2: Word):
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


class Quorum:

    @staticmethod
    def check_synonymy(w1: Word, w2: Word):
        decision1, dist1 = all_sw_models['word2vec_tayga_bow'].check_synonymy(w1, w2)
        decision2, dist2 = all_sw_models['word2vec_araneum_fasttextskipgram_300'].check_synonymy(w1, w2)
        if decision1 and decision2:
            return True
        if decision1 != decision2:
            decision1, sim = all_sw_models['word2vec_tayga_bow'].check_synonymy_by_relative_cosine_similarity(w1, w2)
            decision2, sim = all_sw_models[
                'word2vec_araneum_fasttextskipgram_300'].check_synonymy_by_relative_cosine_similarity(w1, w2)
            if decision1 and decision2:
                return True
            if decision1 != decision2:
                levenshtein = levenshtein_distance(w1.title, w2.title)
                if levenshtein < 3:
                    return True
        return False

# TODO: Предсказатель и эмбеддинги на GPT-2


# TODO: most_similar_to_given - посмотреть, как сделано, реализовать (либо так же, либо через векторную СУБД,
#  либо через какую-то быструю структуру поиска по векторам — на словаре Тайги и на нашем словаре)

# TODO: Посмотреть на most_similar_to_given как на задачу классификации, попробовать k_p_чего-то там для поиска
#  наиболее верояных слов из набора (а не вообще)
#
