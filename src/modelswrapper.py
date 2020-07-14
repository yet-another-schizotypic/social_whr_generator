from sw_core import sw_logger
import sw_constants
import gensim
import os
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import nltk



#TODO: переделать на классы, в которых и модели, и параметры, здесь просто создавать экземпляры

sw_logger.info('Загружаем пакет «stopwords» для nltk')
nltk.download("stopwords")



# sw_logger.info('Начинаем загрузку моделей...')
sw_logger.info('Загружаем модель Word2Vec (tayga, fasttext, B-o-W)')
word2vec_model_file = os.path.join(sw_constants.WORD2VEC_MODEL_PATH, sw_constants.WORD2VEC_MODEL_FILE)
word2vec_wrapper = gensim.models.KeyedVectors.load(word2vec_model_file)

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

sw_logger.info('Все модели загружены успешно!')

# TODO: Предсказатель и эмбеддинги на GPT-2
# TODO: Эмбеддинги на ELMO без Deeppavlov скрипт конвертации: https://github.com/vlarine/transformers-ru
# TODO: самоскачивающиеся BERT'ы из Huggingface

#TODO: модели маленькими буквами (папки)!!!

#TODO: списки / словари моделей и токенайзеров, интерфейсы MODEL и TOKENIZER

#TODO: Загрузка GPT2 и ELMO, подумать над загрузкой on demand