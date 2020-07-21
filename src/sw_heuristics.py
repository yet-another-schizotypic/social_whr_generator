

from sw_constants import SW_SCRIPT_OUTPUT_PATH
import os, json
import itertools
from sw_modelswrapper import all_sw_models
from sw_core import StopTimer, Math, sw_logger, config_parser
from collections import Iterable



class Heuristics:
    # TODO: если что, сделать её рекурсивной для вложенных списков
    def unpack_word_objects_list(self, w_list: list):
        res = []
        for element in w_list:
            res.append(element.title)
        return res

    def pack_heuristic_result_to_string(self, res, hash_sum, target, exp_words):
        return  f'{bool(res)} : {hash_sum} | {target} =?= {exp_words}\n'

    def unpack_string_from_prev_results(self, string):
        s = string.strip(' ').split(' | ')[0]
        prev_res_str = s.split(' : ')[0]
        if prev_res_str == 'True':
            prev_res = True
        elif prev_res_str == 'False':
            prev_res = True
        else:
            prev_res = 'UNK'
        hash_sum_str = s.split(' : ')[1]
        #hash_sum = hash_sum_str.strip("'")
        hash_sum = hash_sum_str
        return prev_res, hash_sum


    def find_canditates_chains_by_processing_elements(self, iterable: Iterable, model_name, stop_timer: StopTimer,
                                                      model_threshold_name: str, func):
        output_dir = config_parser.config['sw_dirs']['output_dir']
        if not os.path.exists(output_dir):
            assert isinstance(output_dir, object)
            os.mkdir(output_dir)

        all_sw_models[model_name].check_init_model_state()
        threshold = all_sw_models[model_name].params[model_threshold_name]

        iterable_info = str(type(iterable)).replace("'",'').replace('<','').replace('>','').replace(' ','_')
        saved_file_name = f'{model_name}-{iterable_info}-{model_threshold_name}={threshold}.txt'
        saved_file = os.path.join(output_dir, saved_file_name)

        prev_results = {}

        if os.path.exists(saved_file):
            for line in open(saved_file, 'r'):
                prev_res, prev_hash = self.unpack_string_from_prev_results(line)
                prev_results[prev_hash] = prev_res

        for element in iterable:
            if stop_timer.check_time_has_gone():
                break

            comb_str = self.unpack_word_objects_list(element)
            hash_sum = Math.get_hash(comb_str)
            if str(hash_sum) in prev_results.keys():
                continue

            target = element[0]
            exp_words = element[1:]
            comb_check_result = func(model_name=model_name, target=target, exp_words=exp_words)
            res_str = self.pack_heuristic_result_to_string(comb_check_result, hash_sum, comb_str[0], comb_str[1:])

            if comb_check_result is True:
                sw_logger.info(f'Найден кандидат! Это строка: {res_str}')
            with open(saved_file, 'a') as fp:
                fp.write(res_str)
            fp.close()

