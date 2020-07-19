# TODO:
# 1. Вынести сюда Math и Word
# 2. Стащить из Permutations генератор псевдо-рандомных выборок
# 3 Сделать очень разные BERT'ы
# 4. В классе графа использовать в атрибутах константы из констант
# 5. Сделать много разныъ бертов, декорировать
# 6. Посмотреть в softmax и около для проверки валидности цепочек
# 7. Класс Word_List с синонимами и произвольной выборкой комбинаций
# 8. Класс chain с проверками валидности
# 9. Добавить разных БЕРТов, отврефакторить имеющийся
# 10. Ембеддинги DistilBert, Elmo

import sw_constants
import logging
import os, pathlib, json


sw_logger = logging.getLogger('socialwhore_loger')
sw_logger.setLevel(sw_constants.LOGLEVEL)
sw_format = logging.Formatter('%(asctime)s - %(message)s')
sw_handler = logging.StreamHandler()
sw_handler.setFormatter(sw_format)
sw_logger.addHandler(sw_handler)

import math
import numpy as np
from gensim import matutils
import scipy
from torch.nn import functional as F
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from itertools import chain

from transformers import top_k_top_p_filtering
from collections import UserDict
import torch


class SWConfigParser:
    def __init__(self, config_file_name=sw_constants.SW_CONFIG_FILE_NAME):
        __project_path__ = pathlib.Path(__file__).parent.absolute()
        config_file = os.path.join(__project_path__, config_file_name)
        with open(config_file, 'r') as fp:
            data = json.load(fp)
        fp.close()
        n_m_flag = False
        sw_supported_models = {}
        for key, value in data.items():
            n_m_flag = False
            if isinstance(value, dict):
                for n, v in value.items():
                    if isinstance(v, list):
                        i = 0
                        for element in v:
                            if os.path.isdir(str('.' + element)) or os.path.isfile(str('.' + element)):
                                v[i] = os.path.join(__project_path__, str('.' + v[i])).replace('/./', '/')
                                i = i + 1
                        sw_supported_models = {**sw_supported_models, **value}
                    elif isinstance(v, dict):
                        for ke, vl in v.items():
                            if os.path.isdir(str('.' + str(vl))) or os.path.isfile(str('.' + str(vl))):
                                v[ke] = os.path.join(__project_path__, str('.' + v[ke])).replace('/./', '/')
                                n_m_flag = True
                        continue

                    elif os.path.isdir(str('.' + v)) or os.path.isfile(str('.' + v)):
                        value[n] = os.path.join(__project_path__, str('.' + v)).replace('/./', '/')
                if n_m_flag is False:
                    sw_supported_models = {**sw_supported_models, **value}

        all_models = {'sw_supported_models': sw_supported_models}
        data = {**data, **all_models}
        self.config = data.copy()


config_parser = SWConfigParser()


# Набор всяких околоматематических штук
class Math:

    @staticmethod
    def get_distance(vec1, vec2):
        return scipy.spatial.distance.cosine(vec1, vec2)

    @staticmethod
    def sum_vectors(*args):
        res: np.ndarray
        i = 0
        for vec in args:
            if i == 0:
                res = np.array(vec)
            else:
                res = np.add(np.array(res), np.array(vec))
            i += 1
        # return matutils.unitvec(res)
        return res





# TODO загрузка циферок для модели из JSON (пока тупо на валидности цепочек)
# TODO почитать про model.eval() и, возможно, заморозить модели или сделать выборку, по которой они будут подбирать константы
# TODO сделать кворум классом
# TODO реализовать проверку на синонимы — cosine и predictions
# TODO реализовать проверку валидности цепочек в моделях и кворуме

""" # Одно слово
class Word:
    __title: str
    word2vec_title_inner: str
    word2vec_embeddings_inner: []
    __elmo_embeddings: []
    __bert_embeddings: []

    def __init__(self, title: str) -> None:
        self.__title = title
        self.__add_noun_suffix()
        self.__word2vec_embeddings = None
        self.__bert_embeddings = None
        self.__elmo_embeddings = None

    def get_word_embeddings(self, embeddings_type=sw_constants.WORD2VEC):
        # TODO: убрать нафиг свойства и присвоение ембеддингов при ините, пихнуть сюда с кэшем
        if embeddings_type == sw_constants.WORD2VEC:
            if self.__word2vec_embeddings is None:
                self.__word2vec_embeddings = NLWrapper.get_Word2Vec_embeddings(self)
            return self.__word2vec_embeddings

        if embeddings_type == sw_constants.BERT:
            if self.__bert_embeddings is None:
                self.__bert_embeddings = NLWrapper.get_BERT_transformer_embeddings(self)
            return self.__bert_embeddings

        if embeddings_type == sw_constants.ELMO:
            if self.__elmo_embeddings is None:
                self.__elmo_embeddings = NLWrapper.get_ELMo_embeddings(self)
            return self.__elmo_embeddings

    def __add_noun_suffix(self, suffix=sw_constants.NOUN_WORD2VEC_SUFFIX):
        self.word2vec_title_inner = self.__title + suffix

    @property
    def title(self):
        return self.__title


# Семантические операции со словами
class NLWrapper:

    @staticmethod
    def get_ELMo_embeddings(word: Word):
        pass
        # TODO: реализация

    @staticmethod
    def get_Word2Vec_embeddings(word: Word):
        res = word2vec_wrapper.get_vector(word.title)
        return res

    @staticmethod
    def get_BERT_transformer_embeddings(word):
        input_ids = conversational_ru_bert_tokenizer.encode(word.title, return_tensors="pt")
        out = conversational_ru_bert_model(input_ids)
        embeddings = out[1][1][:, -1, :].detach().numpy().tolist()
        return embeddings

    @staticmethod
    def __convert_string_sequel_to_lemmatized_list(sequel: str):
        mystem = Mystem()
        russian_stopwords = stopwords.words("russian")
        tokens = mystem.lemmatize(sequel.lower())
        tokens = [token for token in tokens if token not in russian_stopwords \
                  and token != " "
                  and token.strip() not in punctuation]

        tokens = list(set(tokens))
        return tokens

    @staticmethod
    def __creat_clear_sequence_from_list(given_list: list):
        dirty_sequence = str(list(set([item for sublist in given_list for item in sublist])))
        separator = ''
        return separator.join(dirty_sequence).replace('[', '').replace(']', '').replace(',', '').replace("'", '') \

    @staticmethod
    def generate_sequel_for_list_BERT(given_list: list):
        sequence = NLWrapper.__creat_clear_sequence_from_list(given_list)
        input_ids = conversational_ru_bert_tokenizer.encode(sequence, return_tensors="pt")
        next_token_logits = conversational_ru_bert_model(input_ids)[0][:, -1, :]
        # filter
        filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

        # Проходим по разным измерениям в softmax
        sequel = ''
        for dim in range(-2, 2):
            try:
                probs = F.softmax(filtered_next_token_logits, dim=dim)
                next_token = torch.multinomial(probs, num_samples=50)
                sequel = sequel + ' ' + (conversational_ru_bert_tokenizer.decode(next_token[0]))
            except BaseException:
                pass

        translate_table = dict((ord(char), None) for char in punctuation)
        sequel = sequel.translate(translate_table)
        return sequel

    @staticmethod
    def generate_node_candidates_BERT(pairs: list, complete_word_set: list):
        sequel = NLWrapper.generate_sequel_for_list_BERT(pairs)
        # Чистим предложенные варианты
        candidates_list = NLWrapper.__convert_string_sequel_to_lemmatized_list(sequel)
        candidates_list = NLWrapper.__remove_synonyms_from_list(candidates_list, sw_constants.BERT)
        return candidates_list

    @staticmethod
    def is_chain_valid_as_softmax_computation_BERT(target: Word, explanation: list):
        #target_ids = transformers_tokenizer_RU_BERT_CONV.encode(target.title, return_tensors="pt")
        exp_sequence = target.title
        for term in explanation:
            exp_sequence = exp_sequence + ' ' + term.title

        tokenize_input = conversational_ru_bert_tokenizer.tokenize(exp_sequence)
        conversational_ru_bert_model.eval()
        tensor_input = torch.tensor([conversational_ru_bert_tokenizer.convert_tokens_to_ids(tokenize_input)])
        predictions = conversational_ru_bert_model(tensor_input)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions[0].squeeze(), tensor_input.squeeze()).data
        return math.exp(loss)

    @staticmethod
    def __remove_synonyms_from_list(word_list: list, embeddings_type=sw_constants.BERT):
        typedef = sw_constants.ModelSpecificSettings(embeddings_type)
        pass
        # TODO: Этот метод — вообще в WORDLIST
        # TODO: Чистить от синонимов выдачу предсказателя!
        # TODO: чистить предсказатель отдельной процедурой по TD-IDF
        # TODO: чистить предсказатель по количеству букв >=3
        # TODO: унификация предсказателей: строка на входе, строка на выходи, все чистки — потом

        # TODO: math и word — куда-то вынести. BaseUtils? socialwhore-core


# Набор слов (класс Word) в виде словаря {word.title : word}
class WordsDict(UserDict):

    def __delitem__(self, key):
        value = self[key]
        super().__delitem__(key)
        self.pop(value, None)

    def __setitem__(self, key, value):
        if key in self:
            del self[self[key]]
        if value in self:
            del self[value]
        super().__setitem__(key, value)
        super().__setitem__(value, key)

    def __repr__(self):
        return f"{type(self).__name__}({self.data})"

    def add_word(self, word:Word):
        self.__setitem__(word.title, word)

# TODO: загрузка из цепочки / списка


#Цепочка объяснений
class ExplanationChain:
    __target__: Word
    __explanation___: []

    def __init__(self):
        self.__target__ = None
        self.__explanation___ = None

    # def load_from_list(self, chain_list: list):
    #     assert type(chain_list[0]), Word
    #     self.__target__ = chain_list[0]
    #     self.__explanation___ = chain_list[1]

    def load_from_string_list(self, string_chain_list: list):
        self.__target__ = Word(string_chain_list[1][0].lower())
        explanation_list = []
        flat_list = list(chain.from_iterable(string_chain_list[0]))
        for term in flat_list:
            explanation_list.append(Word(term.lower()))
        self.__explanation___ = explanation_list

    def debug_print(self):

        print(self.__target__.title)
        for item in self.__explanation___:
            print(item.title)
        print('---'*100)

    def is_chain_valid_as_explanation_by_mean_distance(self, model_type=sw_constants.WORD2VEC):
        res = sw_constants.ModelSpecificSettings(model_type).clear_embeddings_vec
        for term in self.__explanation___:
            vec = term.get_word_embeddings(model_type)
            res = Math.sum_vectors(res, vec)

        dist = Math.get_distance(self.__target__.get_word_embeddings(model_type), res)
        if dist < sw_constants.ModelSpecificSettings(model_type).min_exp_mean_to_target_cosine_distance:
            return True
        else:
            return False

    def is_chain_valid_as_BERT_check(self):
        res = NLWrapper.is_chain_valid_as_softmax_computation_BERT(self.__target__, self.__explanation___)
        if res < sw_constants.ModelSpecificSettings(sw_constants.BERT).threshold_for_ru_bert_conv_model_chain_validity:
            return True
        else:
            return False

    def test_quorum_decision(self, human_decion: bool):
        w2vec_mean_decision = self.is_chain_valid_as_explanation_by_mean_distance(sw_constants.WORD2VEC)
        bert_mean_decision = self.is_chain_valid_as_explanation_by_mean_distance(sw_constants.BERT)
        bert_loss_decision = self.is_chain_valid_as_BERT_check()

        total_decision_score = w2vec_mean_decision + bert_mean_decision + bert_loss_decision
        if total_decision_score >= 2:
            total_decision = True
        else:
            total_decision = False

        seq = self.__target__.title + ' =?='
        for element in self.__explanation___:
            seq = seq + ' ' + element.title
        seq = seq + ' | w2vmean: ' + str(w2vec_mean_decision)
        seq = seq + ' | bertmean: ' + str(bert_mean_decision)
        seq = seq + ' | bertloss: ' + str(bert_loss_decision)
        seq = seq + ' | QUORUM: ' + str(total_decision)
        seq = seq + ' | human_ref: ' + str(human_decion)
        print(seq)
"""
# TODO: в граф — словарь, чтобы по многу раз не считать векторы
# TODO: сохранение графа — super.save + словарь отдельно, не через pickle, чтобы можно было переписывать код класса
# TODO: models: каждая модель — отдельный класс, начиная с BERT'ов. БЕРТЫ — задекорировать
# TODO: Больше моделей в кворум
# TODO: аналог mostsimilartogiven. Хотя бы просто по векторам, лучше по masked words
# TODO: каждой вершине — простые атрибуты: сколько раз бралась эвристиками в рассмотрение, сколько раз объясняется, в скольки объяснениях участвует
# TODO: Перенести модели отдельно, настроить git.
