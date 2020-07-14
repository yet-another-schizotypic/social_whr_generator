print()
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



sw_logger.info('Начинаем загрузку моделей...')
sw_logger.info('Загружаем модель Word2Vec')
word2vec_model_file = os.path.join(sw_constants.WORD2VEC_MODEL_PATH, sw_constants.WORD2VEC_MODEL_FILE)
word2vec_wrapper = gensim.models.KeyedVectors.load(word2vec_model_file)

sw_logger.info('Загружаем модель RU_BERT_CONV')
transformers_tokenizer_RU_BERT_CONV = AutoTokenizer.from_pretrained(sw_constants.BERT_MODEL_PATH)
transformers_from_RU_BERT_CONV_model = AutoModelWithLMHead.from_pretrained(sw_constants.BERT_MODEL_PATH)

sw_logger.info('Все модели загружены успешно!')

# TODO: Предсказатель и эмбеддинги на GPT-2
