from sw_constants import SW_SCRIPT_OUTPUT_PATH
import os, json
import itertools
from sw_modelswrapper import all_sw_models
from sw_core import StopTimer, Math, sw_logger



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

    # TODO — в декоратор!
    def find_canditates_chains_by_simple_cosmul(self, word_list, model_name):
        # TODO брать его из конфига
        output_dir = os.path.join(SW_SCRIPT_OUTPUT_PATH, 'heuristics/find_canditates_chains_by_simple_cosmul/')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        saved_file = os.path.join(output_dir, 'results_001.txt')

        prev_results = {}

        if os.path.exists(saved_file):
            for line in open(saved_file, 'r'):
                prev_res, prev_hash = self.unpack_string_from_prev_results(line)
                prev_results[prev_hash] = prev_res

        all_sw_models[model_name].check_init_model_state()

        combinations = itertools.combinations(word_list, 6)
        stop_timer = StopTimer(end_time="19:55:00", tick="00:05:00")
        #TODO: параметры таймера из кода в вызов


        for comb in combinations:
            if stop_timer.check_time_has_gone():
                break

            comb_str = self.unpack_word_objects_list(comb)
            hash_sum = Math.get_hash(comb_str, 10)

            if str(hash_sum) in prev_results.keys():
                continue

            target = comb[0]
            exp_words = comb[1:]
            comb_check_result = all_sw_models[model_name].check_explanation_chain_validity(target, exp_words)
            res_str = self.pack_heuristic_result_to_string(comb_check_result, hash_sum, comb_str[0], comb_str[1:])
            if comb_check_result is True:
                sw_logger.info(f'Найден кандидат! Это строка: {res_str}')
            with open(saved_file, 'a') as fp:
                fp.write(res_str)
            fp.close()

