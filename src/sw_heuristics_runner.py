from sw_graphs import WordGraph
import sw_constants
from sw_heuristics import Heuristics
import os
from sw_core import StopTimer
import itertools
from sw_modelswrapper import all_sw_models


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
    heur.find_canditates_chains_by_processing_elements(iterable=combinations, model_name='word2vec_tayga_bow',
                                                   stop_timer=stop_timer,
                                                   model_threshold_name='cosmul_similarity_for_chain_validation',
                                                   func=check_chain_validity_with_given_model)

def check_by_random_samples(sample_count_per_one_run=1000000):
    fname = os.path.join(sw_constants.SW_SCRIPT_PATH, 'real_data_by_humans.txt')
    wg = WordGraph()
    wg.initialize_from_file(fname)
    word_list = wg.get_random_samples_chains(min_len=5, max_len=8, count=sample_count_per_one_run)
    stop_timer = StopTimer(end_time="14:45:00", tick="00:05:00")
    heur = Heuristics()
    heur.find_canditates_chains_by_processing_elements(iterable=word_list, model_name='bert_base_multilingual_cased',
                                                       stop_timer=stop_timer,
                                                       model_threshold_name='exp_loss_similarity_for_chain_validation_max',
                                                       func=check_chain_validity_with_given_model)


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

run = check_by_random_samples(1000000)

#run = check_bert_with_permutations(10000)