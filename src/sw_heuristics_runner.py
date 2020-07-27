import csv

from sw_graphs import WordGraph
import sw_constants
from sw_heuristics import Heuristics
from sw_core import SWUtils, config_parser, ProgressBar
import os
from sw_core import StopTimer, Math
import itertools
from sw_modelswrapper import all_sw_models, Word


def check_chain_validity_with_given_model(model_name, target, exp_words):
    return all_sw_models[model_name].check_explanation_chain_validity(target, exp_words)


def check_explanation_chain_validity_with_permutations(model_name, target, exp_words):
    return all_sw_models[model_name].check_explanation_chain_validity_with_permutations(target, exp_words)


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
    #TODO: data_dir — вынести в конфиг
    output_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
    output_dir = os.path.join(output_dir, 'pipeline/')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_file = os.path.join(output_dir, 'big_quasi_random_chains_file.csv')


    vocab_file_name = os.path.join(sw_constants.SW_SCRIPT_DATA_PATH, 'united_dict.txt')

    word_list = SWUtils.read_vocab_without_duplicates(vocab_file_name, check_synonymy=False)

    chain_list = SWUtils.generate_quasi_random_samples_from_string_list(string_list=word_list, min_len=6,
                                                                        max_len=6, count=sample_count_per_one_run)
    total_collisions = 0
    pb = ProgressBar(total=sample_count_per_one_run)
    for chain in chain_list:
        hash_sum = Math.get_hash(str(chain))
        if hash_sum in used_hashes.keys():
            total_collisions += 1
            continue
        used_hashes[hash_sum] = True

        with open(output_file, 'a') as fp:
            writer = csv.writer(fp, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow([hash_sum, chain[0], str(chain[1:]), 'random'])
        fp.close()
        pb.print_progress_bar()
    print(f'Процесс завершен, коллизий: {total_collisions}')


def do_improve_chains(batch_size=1000):
    f_h_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
    input_dir = os.path.join(f_h_dir, 'pipeline')
    input_file = os.path.join(input_dir, 'big_quasi_random_chains_file.csv')
    vocab_file_name = os.path.join(sw_constants.SW_SCRIPT_DATA_PATH, 'united_dict.txt')

    vocab_words_list = SWUtils.read_vocab_without_duplicates(vocab_file_name, check_synonymy=False)

    Heuristics.improve_chains(chains_file=input_file, model_name='word2vec_tayga_bow', vocabulary=vocab_words_list, total_improvements=batch_size)

#run = do_improve_chains(batch_size=10000)

#TODO: протестировать буфферы на экселе

#run = produce_append_big_file_for_model_tests(10000000)

run = Heuristics.do_precomputations_by_file(['elmo_tayga_lemmas_2048'], True)

#run = produce_append_big_file_for_model_tests(1000000)

#run = Heuristics.do_precomputations_by_file([], True)

# TODO: другие БЕРТы, ELMo, XLNET
# TODO: найти табличную альтернативу экселю под MacOS
# TODO: В таймер добавить вывод скорости per 1000, например или per minute
# TODO: Semantic similarity отсюда в word2vec: https://habr.com/ru/post/275913/
