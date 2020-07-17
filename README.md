# social_whr_generator

# __project_path__ = pathlib.Path(__file__).parent.parent.absolute()
# # ===Основные константы===
#
# #==Модели на основе архитектуры BERT==
# # В конфигурационный файл этоих модели нужно добавить строчку "output_hidden_states": "True", если её там нет
# #Conversational RuBERT, http://docs.deeppavlov.ai/en/master/features/models/bert.html
# __conversational_ru_bert_model_relative_path__ = 'models/BERT/conversational_ru_bert/ru_conversational_cased_L-12_H-768_A-12_pt'
# CONVERSATIONAL_RU_BERT_MODEL_PATH = os.path.join(__project_path__, __conversational_ru_bert_model_relative_path__)
#
#
# #RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters, http://docs.deeppavlov.ai/en/master/features/models/bert.html
# __ru_bert_cased_model_relative_path__ = 'models/BERT/ru_bert_cased/ru_bert_cased_l-12_H-768_A-12_pt/'
# RU_BERT_CASED_MODEL_PATH = os.path.join(__project_path__, __ru_bert_cased_model_relative_path__)
#
# #Sentence RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M, http://docs.deeppavlov.ai/en/master/features/models/bert.html
# __sentence_ru_bert_model_relative_path__ = 'models/BERT/sentence_ru_bert/sentence_ru_cased_L-12_H-768_A-12_pt/'
# SENTENCE_RU_BERT_MODEL_PATH = os.path.join(__project_path__, __sentence_ru_bert_model_relative_path__)
#
# #Slavic BERT от deeppavlov, http://docs.deeppavlov.ai/en/master/features/models/bert.html
# __slavic_bert_model_relative_path__ = 'models/BERT/slavic_bert/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt/'
# SLAVIC_BERT_MODEL_PATH = os.path.join(__project_path__, __slavic_bert_model_relative_path__)
#
# # bert-base-multilingual-uncased из Transformers, https://huggingface.co/bert-base-multilingual-uncased#list-files
# __bert_base_multilingual_uncased_model_relative_path__ = 'models/BERT/bert_base_multilingual_uncased/'
# BERT_BASE_MULTILINGUAL_UNCASED_MODEL_PATH = os.path.join(__project_path__, __bert_base_multilingual_uncased_model_relative_path__)
#
# # bert-base-multilingual-cased из Transformers, https://huggingface.co/bert-base-multilingual-cased#list-files
# __bert_base_multilingual_cased_model_relative_path__ = 'models/BERT/bert_base_multilingual_cased/'
# BERT_BASE_MULTILINGUAL_CASED_MODEL_PATH = os.path.join(__project_path__, __bert_base_multilingual_cased_model_relative_path__)
#
#
# #==Модель на основе архитектуры GPT, https://github.com/vlomme/Russian-gpt-2
# # Как получить:
# # Скачиваем с гитхаба файл download_model.py, c он качает модель. Но она — в Tensoflow, а у наст
# # всё на pytorch. Поэтому нужно сконвертировать. Для этого:
# # 1. Устанавливаем пакет transformers;
# # 2. Делаем грязный хак: в файле convert_gpt2_checkpoint_to_pytorch.py — меняем конструкцию:
# #
# # Construct model
# # if gpt2_config_file == "":
# #     config = GPT2Config()
# # else:
# #     config = GPT2Config.from_json_file(gpt2_config_file)
# # model = GPT2Model(config)
# #
# #на:
# #config = GPT2Config.from_pretrained('gpt2-medium')  # Replace 'gpt2-medium' with whichever model spec you're converting
# #model = GPT2Model(config)
# #
# #Далее — с помощью утилиты transformers-cli convert производим конвертацию.
# #Документация — тут: https://huggingface.co/transformers/converting_tensorflow_models.html
# #Далее переименовываем файлы:
# # mv encoder.json2 vocab.json
# # mv vocab2.bpe merges.txt
# __gpt2_model_relative_path__ = 'models/GPT2/Russian-gpt-2-finetuning/'
# GPT2_MODEL_PATH = os.path.join(__project_path__, __gpt2_model_relative_path__)
#
# #ELMo, tayga_lemmas_elmo_2048_2019, https://rusvectores.org/ru/models/
# __elmo_model_relative_path__ = 'models/ELMo/tayga_lemmas_elmo_2048_2019/'
# __elmo_model_path___ = os.path.join(__project_path__, __elmo_model_relative_path__)
# ELMO_MODEL_OPTIONS_FILE = os.path.join(__elmo_model_path___, 'options.json')
# ELMO_MODEL_WEIGHTS_FILE = os.path.join(__elmo_model_path___, 'model.hdf5')
#
# #==Пути до файлов модели Word2vec fasttext B-o-W==
# #Файл можно взять тут: https://rusvectores.org/ru/models/
# __word2vec_model_relative_path__ = 'models/word2vec/tayga_none_fasttextcbow_300_10_2019'
# WORD2VEC_MODEL_PATH = os.path.join(__project_path__, __word2vec_model_relative_path__)
# WORD2VEC_MODEL_FILE = os.path.join(WORD2VEC_MODEL_PATH, 'model.model')
#
# WORD2VEC_TAYGA_BOW_NAME = 'WORD2VEC_TAYGA_BOW'
# CONVERSATIONAL_RU_BERT_NAME = 'CONVERSATIONAL_RU_BERT'
# RU_BERT_CASED_NAME = 'RU_BERT_CASED'
# SENTENCE_RU_BERT_NAME = 'SENTENCE_RU_BERT'
# SLAVIC_BERT_NAME = 'SLAVIC_BERT_MODEL'
# BERT_BASE_MULTILINGUAL_UNCASED_NAME = 'BERT_BASE_MULTILINGUAL_UNCASED'
# BERT_BASE_MULTILINGUAL_CASED_NAME = 'BERT_BASE_MULTILINGUAL_CASED'
# ELMO_TAYGA_LEMMAS_2048_NAME = 'ELMO_TAYGA_LEMMAS_2048'
# GPT2_RUSSIAN_FINETUNING_NAME = 'GPT2_RUSSIAN_FINETUNING'
#
# # TODO: почитать про docstring и тройные кавычки
#
# SW_BERT_MODELS = {BERT_BASE_MULTILINGUAL_UNCASED_NAME: BERT_BASE_MULTILINGUAL_UNCASED_MODEL_PATH,
#                   BERT_BASE_MULTILINGUAL_CASED_NAME: BERT_BASE_MULTILINGUAL_CASED_MODEL_PATH,
#                   CONVERSATIONAL_RU_BERT_NAME: CONVERSATIONAL_RU_BERT_MODEL_PATH, RU_BERT_CASED_NAME: RU_BERT_CASED_MODEL_PATH,
#                   SENTENCE_RU_BERT_NAME: SENTENCE_RU_BERT_MODEL_PATH, SLAVIC_BERT_NAME: SLAVIC_BERT_MODEL_PATH}
#
# SW_WORD2VEC_MODELS = {WORD2VEC_TAYGA_BOW_NAME: WORD2VEC_MODEL_FILE}
#
# SW_SUPPORTED_MODELS = {**SW_BERT_MODELS, **SW_WORD2VEC_MODELS}
#
# SW_ELMO_MODELS = {ELMO_TAYGA_LEMMAS_2048_NAME: [ELMO_MODEL_OPTIONS_FILE, ELMO_MODEL_WEIGHTS_FILE]}
#
# SW_SUPPORTED_MODELS = {**SW_SUPPORTED_MODELS, **SW_ELMO_MODELS}
#
# SW_GPT2_MODELS = {GPT2_RUSSIAN_FINETUNING_NAME: GPT2_MODEL_PATH}
#
# SW_SUPPORTED_MODELS = {**SW_SUPPORTED_MODELS, **SW_GPT2_MODELS}

# NOUN_WORD2VEC_SUFFIX = '_NOUN'
#
# MAX_NEIGHBOUR_COSINE_DISTANCE_WORD2VEC = 0.650
# MIN_NEIGHBOUR_COSINE_DISTANCE_WORD2VEC = 0.320
# # Среднее гармоническое на «неправильной» выборке — 0.909286. Ниже — значение из «правильной».
# MIN_EXP_MEAN_TO_TARGET_COSINE_DISTANCE_WORD2VEC = 0.571643
#
# MAX_NEIGHBOUR_COSINE_DISTANCE_BERT = 0.07
# MIN_NEIGHBOUR_COSINE_DISTANCE_BERT = 0.02
# # Среднее гармоническое на «неправильной» выборке — 0,019015
# MIN_EXP_MEAN_TO_TARGET_COSINE_DISTANCE_BERT = 0.015532
#
# # Метрика для БЕРТА, тест, переделать. Среднее арифметическое по «правильной» выборке 611,948501
# THRESHOLD_FOR_RU_BERT_CONV_MODEL_CHAIN_VALIDITY = 612
#
#
# class ModelSpecificSettings:
#     data_type_for_nx_graph: str
#     clear_embeddings_vec: []
#     max_cosine_neighbourhood_distance: float
#     min_cosine_neighbourhood_distance: float
#     min_exp_mean_to_target_cosine_distance: float
#     threshold_for_ru_bert_conv_model_chain_validity: float
#
#     def __init__(self, model_type: str):
#
#         if model_type == WORD2VEC:
#             self.data_type_for_nx_graph = 'word2vec_embeddings'
#             self.clear_embeddings_vec = [0] * 300
#             self.max_cosine_neighbourhood_distance = MAX_NEIGHBOUR_COSINE_DISTANCE_WORD2VEC
#             self.min_cosine_neighbourhood_distance = MIN_NEIGHBOUR_COSINE_DISTANCE_WORD2VEC
#             self.min_exp_mean_to_target_cosine_distance = MIN_EXP_MEAN_TO_TARGET_COSINE_DISTANCE_WORD2VEC
#
#         if model_type == BERT:
#             self.data_type_for_nx_graph = 'bert_embeddings'
#             self.clear_embeddings_vec = [0] * 768
#             self.max_cosine_neighbourhood_distance = MAX_NEIGHBOUR_COSINE_DISTANCE_BERT
#             self.min_cosine_neighbourhood_distance = MIN_NEIGHBOUR_COSINE_DISTANCE_BERT
#             self.min_exp_mean_to_target_cosine_distance = MIN_EXP_MEAN_TO_TARGET_COSINE_DISTANCE_BERT
#             self.threshold_for_ru_bert_conv_model_chain_validity = THRESHOLD_FOR_RU_BERT_CONV_MODEL_CHAIN_VALIDITY


# import json
#
# dict = {}
# dict['version'] = '0.000'
# dict['sw_bert_models'] = SW_BERT_MODELS
# dict['sw_word2vec_models'] = SW_WORD2VEC_MODELS
# dict['sw_elmo_modles'] = SW_ELMO_MODELS
# dict['sw_gpt2_models'] = SW_GPT2_MODELS
#
# with open('/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/src/sw_config.json', 'w') as fp:
#     json.dump(dict, fp, sort_keys=True, indent=4)
# fp.close()
