from sw_graphs import WordGraph
import sw_constants
from sw_heuristics import Heuristics
import os
from sw_core import StopTimer, Math
import itertools
from sw_modelswrapper import all_sw_models, nl_wrapper


def check_chain_validity_with_given_model(model_name, target, exp_words):
    return all_sw_models[model_name].check_explanation_chain_validity(target, exp_words)


def check_explanation_chain_validity_with_permutations(model_name, target, exp_words):
    return all_sw_models[model_name].check_explanation_chain_validity_with_permutations(target, exp_words)


def check_by_combinations():
    fname = os.path.join(sw_constants.SW_SCRIPT_PATH, 'real_data_by_humans.txt')
    wg = WordGraph()
    wg.initialize_from_file(fname)
    word_list = wg.get_all_words_from_dict()
    combinations = itertools.combinations(word_list, 6)
    stop_timer = StopTimer(end_time="23:51:00", tick="00:02:00")
    heur = Heuristics()
    heur.find_canditates_chains_by_processing_elements(iterable=combinations, model_name='m_russian_gpt2-aws',
                                                       stop_timer=stop_timer,
                                                       model_threshold_min='exp_loss_similarity_for_chain_validation_min',
                                                       model_threshold_max='exp_loss_similarity_for_chain_validation_max',
                                                       func=check_chain_validity_with_given_model)


def check_by_random_samples(sample_count_per_one_run=1000000):
    fname = os.path.join(sw_constants.SW_SCRIPT_PATH, 'real_data_by_humans.txt')
    wg = WordGraph()
    wg.initialize_from_file(fname)
    word_list = wg.get_random_samples_chains(min_len=5, max_len=8, count=sample_count_per_one_run)
    stop_timer = StopTimer(end_time="16:50:00", tick="00:05:00")
    heur = Heuristics()
    heur.find_canditates_chains_by_processing_elements(iterable=word_list, model_name='bert_base_multilingual_cased',
                                                       stop_timer=stop_timer,
                                                       model_threshold_min='exp_loss_similarity_for_chain_validation_min',
                                                       model_threshold_max='exp_loss_similarity_for_chain_validation_max',
                                                       type_prefix='mv',
                                                       func=check_chain_validity_with_given_model)


# def generate_large_text_file_with_chains_one_batch(min_len=6, max_len=8, sample_count_per_one_run=1000):
#     fname = os.path.join(sw_constants.SW_SCRIPT_PATH, 'real_data_by_humans.txt')
#     wg = WordGraph()
#     wg.initialize_from_file(fname)
#     chains_list = wg.get_random_samples_chains(min_len=min_len, max_len=max_len, count=sample_count_per_one_run)
#     dict = {}
#     for chain in chains_list:
#         распоковываем цепочку
#         считаем хэш
#         Пишем в словарь
#         Добавляем словам атрибуты


# def check_bert_with_permutations(sample_count_per_one_run=1000000):
#     fname = os.path.join(sw_constants.SW_SCRIPT_PATH, 'real_data_by_humans.txt')
#     wg = WordGraph()
#     wg.initialize_from_file(fname)
#     word_list = wg.get_random_samples_chains(min_len=5, max_len=8, count=sample_count_per_one_run)
#     stop_timer = StopTimer(end_time="14:45:00", tick="00:00:30")
#     heur = Heuristics()
#     heur.find_canditates_chains_by_processing_elements(iterable=word_list, model_name='conversational_ru_bert',
#                                                        stop_timer=stop_timer,
#                                                        model_threshold_name='exp_loss_similarity_for_chain_validation_max',
#                                                        func=check_explanation_chain_validity_with_permutations)

#run = check_by_random_samples(1000000)
# run = check_by_combinations()

# run = check_bert_with_permutations(10000)

def produce_append_big_file_for_model_tests(sample_count_per_one_run):
    fname = os.path.join(sw_constants.SW_SCRIPT_PATH, 'real_data_by_humans.txt')
    wg = WordGraph()
    wg.initialize_from_file(fname)
    chain_list = wg.get_random_samples_chains(min_len=6, max_len=6, count=sample_count_per_one_run)
    Heuristics.create_file_for_precomputations(chain_list)

#run = produce_append_big_file_for_model_tests(10000000)

#run = Heuristics.do_precomputations_by_file(['elmo_tayga_lemmas_2048'], True)
run = Heuristics.do_precomputations_by_file([], True)

# TODO: другие БЕРТы, ELMo, XLNET
# TODO: найти табличную альтернативу экселю под MacOS
# TODO: В таймер добавить вывод скорости per 1000, например или per minute
# TODO: Semantic similarity отсюда в word2vec: https://habr.com/ru/post/275913/
