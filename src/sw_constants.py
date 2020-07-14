print()
import logging

LOGLEVEL = logging.INFO

# Основные константы
WORD2VEC_MODEL_PATH = '//Word2vec/models/tayga_none_fasttextcbow_300_10_2019'
WORD2VEC_MODEL_FILE = 'model.model'
# В конфиг модели BERT добавить строчку "output_hidden_states": "True", если её там нет
BERT_MODEL_PATH = '//bert/models/deeppavlov/conversational_RuBERT_tf/ru_conversational_cased_L-12_H-768_A-12'

WORD2VEC = 'word2vec'
BERT = 'bert'
ELMO = 'ELMo'

NOUN_WORD2VEC_SUFFIX = '_NOUN'

MAX_NEIGHBOUR_COSINE_DISTANCE_WORD2VEC = 0.650
MIN_NEIGHBOUR_COSINE_DISTANCE_WORD2VEC = 0.320
# Среднее гармоническое на «неправильной» выборке — 0.909286. Ниже — значение из «правильной».
MIN_EXP_MEAN_TO_TARGET_COSINE_DISTANCE_WORD2VEC = 0.571643

MAX_NEIGHBOUR_COSINE_DISTANCE_BERT = 0.07
MIN_NEIGHBOUR_COSINE_DISTANCE_BERT = 0.02
# Среднее гармоническое на «неправильной» выборке — 0,019015
MIN_EXP_MEAN_TO_TARGET_COSINE_DISTANCE_BERT = 0.015532

# Метрика для БЕРТА, тест, переделать. Среднее арифметическое по «правильной» выборке 611,948501
THRESHOLD_FOR_RU_BERT_CONV_MODEL_CHAIN_VALIDITY = 612


class ModelSpecificSettings:
    data_type_for_nx_graph: str
    clear_embeddings_vec: []
    max_cosine_neighbourhood_distance: float
    min_cosine_neighbourhood_distance: float
    min_exp_mean_to_target_cosine_distance: float
    threshold_for_ru_bert_conv_model_chain_validity: float

    def __init__(self, model_type: str):

        if model_type == WORD2VEC:
            self.data_type_for_nx_graph = 'word2vec_embeddings'
            self.clear_embeddings_vec = [0] * 300
            self.max_cosine_neighbourhood_distance = MAX_NEIGHBOUR_COSINE_DISTANCE_WORD2VEC
            self.min_cosine_neighbourhood_distance = MIN_NEIGHBOUR_COSINE_DISTANCE_WORD2VEC
            self.min_exp_mean_to_target_cosine_distance = MIN_EXP_MEAN_TO_TARGET_COSINE_DISTANCE_WORD2VEC

        if model_type == BERT:
            self.data_type_for_nx_graph = 'bert_embeddings'
            self.clear_embeddings_vec = [0] * 768
            self.max_cosine_neighbourhood_distance = MAX_NEIGHBOUR_COSINE_DISTANCE_BERT
            self.min_cosine_neighbourhood_distance = MIN_NEIGHBOUR_COSINE_DISTANCE_BERT
            self.min_exp_mean_to_target_cosine_distance = MIN_EXP_MEAN_TO_TARGET_COSINE_DISTANCE_BERT
            self.threshold_for_ru_bert_conv_model_chain_validity = THRESHOLD_FOR_RU_BERT_CONV_MODEL_CHAIN_VALIDITY
