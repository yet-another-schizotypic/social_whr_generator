import operator

from sw_constants import SW_SCRIPT_OUTPUT_PATH
import os, json
import itertools
from sw_modelswrapper import all_sw_models, nl_wrapper, Word
from sw_core import StopTimer, Math, sw_logger, config_parser, ProgressBar
from collections import Iterable


class Heuristics:

    @staticmethod
    def pack_heuristic_result_to_string(res, hash_sum, threshold_min, threshold_max,
                                        type_prefix, model_name, metric_res, target,
                                        exp_words):
        return f'{bool(res)} : {hash_sum} : {threshold_min} : {threshold_max} : {type_prefix} : {model_name} : {metric_res} | {target} =?= {exp_words}\n'

    @staticmethod
    def unpack_string_from_prev_results(string):
        s = string.strip(' ').split(' | ')[0]
        prev_res_str = s.split(' : ')[0]
        if prev_res_str == 'True':
            prev_res = True
        elif prev_res_str == 'False':
            prev_res = True
        else:
            prev_res = 'UNK'
        results_1 = s.split(' : ')
        hash_sum_str = results_1[1]
        # hash_sum = hash_sum_str.strip("'")
        hash_sum = hash_sum_str
        threshold_min_str = results_1[2]
        threshold_max_str = results_1[3]
        type_prefix_str = results_1[4]
        model_name_str = results_1[5]
        metric_res = results_1[6]

        s = string.strip(' ').split(' | ')[1]
        results_2 = s.split(' =?= ')
        target = results_2[0]
        exp_words = results_2[1].replace("[", '').replace("'", '').replace(',', '').replace(']', '').replace('\n', '')
        exp_words = exp_words.split(' ')
        return prev_res, hash_sum, threshold_min_str, threshold_max_str, type_prefix_str, model_name_str, metric_res, target, exp_words

    @staticmethod
    def find_canditates_chains_by_processing_elements(iterable: Iterable, model_name, stop_timer: StopTimer,
                                                      model_threshold_min: str, model_threshold_max: str,
                                                      type_prefix: str, func):
        output_dir = config_parser.config['sw_dirs']['output_dir']
        if not os.path.exists(output_dir):
            assert isinstance(output_dir, object)
            os.mkdir(output_dir)

        all_sw_models[model_name].check_init_model_state()
        threshold_min = all_sw_models[model_name].params[model_threshold_min]
        threshold_max = all_sw_models[model_name].params[model_threshold_max]

        iterable_info = str(type(iterable)).replace("'", '').replace('<', '').replace('>', '').replace(' ', '_')
        saved_file_name = f'{type_prefix}_{model_name}-{iterable_info}-{model_threshold_min}={threshold_min}-{model_threshold_max}={threshold_max}.txt'
        saved_file = os.path.join(output_dir, saved_file_name)

        prev_results = {}

        if os.path.exists(saved_file):
            for line in open(saved_file, 'r'):
                prev_res, prev_hash, _, _, _, _, _, _, _ = self.unpack_string_from_prev_results(line)
                prev_results[prev_hash] = prev_res

        for element in iterable:
            if stop_timer.check_time_has_gone():
                break

            comb_str = nl_wrapper.unpack_word_objects_list(element)
            hash_sum = Math.get_hash(comb_str)
            if str(hash_sum) in prev_results.keys():
                continue

            target = element[0]
            exp_words = element[1:]
            comb_check_result, metric_res = func(model_name=model_name, target=target, exp_words=exp_words)
            res_str = self.pack_heuristic_result_to_string(comb_check_result, hash_sum, threshold_min, threshold_max,
                                                           type_prefix, model_name,
                                                           metric_res, comb_str[0], comb_str[1:])

            if comb_check_result is True:
                sw_logger.info(f'Найден кандидат! Это строка: {res_str}')
            with open(saved_file, 'a') as fp:
                fp.write(res_str)
            fp.close()

    @staticmethod
    def create_file_for_precomputations(chain_list):
        output_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        pass

        hashes = {}
        out_file_name = os.path.join(output_dir, "generated_big_file.txt")
        if os.path.exists(out_file_name):
            for line in open(out_file_name, 'r'):
                _, has_sum, _, _, _, _, _, _, _ = Heuristics.unpack_string_from_prev_results(line)
                hashes[has_sum] = True

        pb = ProgressBar(total=len(chain_list), epoch_length=(len(chain_list) // 1000))
        total_added = 0
        total_collisions = 0
        added_hashes = {}
        for element in chain_list:
            pb.print_progress_bar()
            chain = nl_wrapper.unpack_word_objects_list(element)

            hash_sum = str(Math.get_hash(chain))
            if (hash_sum in hashes.keys()) or (hash_sum in added_hashes.keys()):
                total_collisions = total_collisions + 1
                continue
            added_hashes[hash_sum] = True
            prev_res = False
            total_added = total_added + 1
            threshold_min_str = "UNK"
            threshold_max_str = "UNK"
            type_prefix_str = "fg"
            model_name_str = "NOT DEFINED"
            metric_res = "N/A"
            target = chain[0]
            exp_words = chain[1:]
            fin_str = Heuristics.pack_heuristic_result_to_string(prev_res, hash_sum, threshold_min_str,
                                                                 threshold_max_str,
                                                                 type_prefix_str, model_name_str, metric_res, target,
                                                                 exp_words)
            with open(out_file_name, 'a') as fp:
                fp.write(fin_str)
            fp.close()

        print(
            f'Кривой рандом допустил {total_collisions} совпадений, поэтому реально добавили только {total_added} записей')

    @staticmethod
    def file_len(file_name):
        with open(file_name) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    @staticmethod
    def do_precomputations_by_file(unsupported_models: list, do_equity: bool, next_step_count=100000):
        f_h_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
        out_dir = os.path.join(f_h_dir, 'models_output/')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        do_next_step = True

        while do_next_step is True:
            models_statuses = {}
            models_output_files = {}
            for name, model in all_sw_models.items():
                if not (name in unsupported_models):
                    model_output_file = f'{name}_fh_out.txt'
                    model_output_file = os.path.join(out_dir, model_output_file)
                    models_output_files[name] = model_output_file
                    if os.path.exists(model_output_file):
                        models_statuses[name] = Heuristics.file_len(model_output_file)
                    else:
                        models_statuses[name] = 0

            max_precomputations = max(models_statuses.items(), key=operator.itemgetter(1))
            min_precomputations = min(models_statuses.items(), key=operator.itemgetter(1))
            print(f'Наибольшее количество цепочек ({max_precomputations[1]}) обработала {max_precomputations[0]}')
            print(f'Меньше всех цепочек ({min_precomputations[1]}) обработала {min_precomputations[0]}')
            max_diff = max_precomputations[1] - min_precomputations[1]

            model_name = min_precomputations[0]
            if max_diff == 0:
                if (max_precomputations[1] == 0) or do_equity is False:
                    max_diff = next_step_count
                    sw_logger.info(f'Работать, негры! Сейчас {model_name} будет делать ещё {max_diff} цепочек!')
                elif do_equity is True:
                    do_next_step = False
                    sw_logger.info(f'Свобода, равенство, браство! Все модели обработали {max_precomputations[1]} цепочек.')
                    break

            pb = ProgressBar(total=max_diff, epoch_length=(max_diff // 500))

            sw_logger.info(f'Сейчас нужно, чтобы {model_name} обработала {max_diff} цепочек.')
            model_output_file = models_output_files[model_name]
            hashes = {}

            i = 0
            # читаем хэши уже проверенных цепочек в словарь, чтобы не повторяться в вычислениях
            if os.path.exists(model_output_file):
                for line in open(model_output_file, 'r'):
                    _, has_sum, _, _, _, _, _, _, _ = Heuristics.unpack_string_from_prev_results(line)
                    hashes[has_sum] = True

            # начинаем читать цепочки из файла
            input_file_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
            input_file_name = os.path.join(input_file_dir, "generated_big_file.txt")
            for line in open(input_file_name, 'r'):

                #Если добрали разницу — выходим
                if i >= max_diff:
                    break

                prev_res_str, hash_sum_str, threshold_min_str, threshold_max_str, \
                type_prefix_str, model_name_str, \
                metric_res, target, exp_words = Heuristics.unpack_string_from_prev_results(line)

                # Убеждаемся, что такую мы ещё не считали
                if hash_sum_str in hashes.keys():
                    continue


                i = i + 1

                # Проверяем цепочку
                target_w = Word(target)
                exp_list = [Word(word) for word in exp_words]
                chain_validity, metric_res = all_sw_models[model_name].check_explanation_chain_validity(target_w, exp_list)
                pb.print_progress_bar()
                # Формируем строку для записи
                # определяем имя граничных параметров в зависимости от моделями
                if 'word2vec' in model_name:
                    t_min_name = 'cosmul_similarity_for_chain_validation_min'
                    t_max_name = 'cosmul_similarity_for_chain_validation'
                if 'bert' in model_name or 'gpt2' in model_name:
                    t_min_name = 'exp_loss_similarity_for_chain_validation_min'
                    t_max_name = 'exp_loss_similarity_for_chain_validation_max'
                if 'elmo' in model_name:
                    t_min_name = 'vec_similarity_for_chain_validation_min'
                    t_max_name = 'vec_similarity_for_chain_validation_max'
                if 'xlnet' in model_name:
                    t_min_name = 'not_implemented_in_sw'
                    t_max_name = 'not_implemented_in_sw'

                threshold_min = all_sw_models[model_name].params[t_min_name]
                threshold_max = all_sw_models[model_name].params[t_max_name]
                type_prefix = "fv"

                res_str = Heuristics.pack_heuristic_result_to_string(chain_validity, hash_sum_str, threshold_min,
                                                                     threshold_max,
                                                                     type_prefix, model_name,
                                                                     metric_res, target, exp_words)
                if chain_validity is True:
                    sw_logger.info(f'Найден кандидат! Это строка: {res_str}')
                with open(model_output_file, 'a') as fp:
                    fp.write(res_str)
                fp.close()
