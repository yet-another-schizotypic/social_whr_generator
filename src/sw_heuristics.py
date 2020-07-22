from sw_constants import SW_SCRIPT_OUTPUT_PATH
import os, json
import itertools
from sw_modelswrapper import all_sw_models, nl_wrapper
from sw_core import StopTimer, Math, sw_logger, config_parser
from collections import Iterable


class Heuristics:

    def pack_heuristic_result_to_string(self, res, hash_sum, threshold_min, threshold_max,
                                        type_prefix, model_name, metric_res, target,
                                        exp_words):
        return f'{bool(res)} : {hash_sum} : {threshold_min} : {threshold_max} : {type_prefix} : {model_name} : {metric_res} | {target} =?= {exp_words}\n'

    def unpack_string_from_prev_results(self, string):
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
        exp_words = results_2[1]
        return prev_res, hash_sum, threshold_min_str, threshold_max_str, type_prefix_str, model_name_str, metric_res, target, exp_words

    def find_canditates_chains_by_processing_elements(self, iterable: Iterable, model_name, stop_timer: StopTimer,
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
                prev_res, prev_hash, _, _, _, _, _ = self.unpack_string_from_prev_results(line)
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
