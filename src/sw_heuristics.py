import csv
import operator
import re

from sw_constants import SW_SCRIPT_OUTPUT_PATH
import os, json
import itertools
from sw_modelswrapper import all_sw_models, nl_wrapper, Word
from sw_core import StopTimer, Math, sw_logger, config_parser, ProgressBar, SWUtils, CSVWriter, CSVReader
from collections.abc import Iterable


class Heuristics:

    # DEPRECATED: удалить через пару версий
    # @staticmethod
    # def pack_heuristic_result_to_string(res, hash_sum, threshold_min, threshold_max,
    #                                     type_prefix, model_name, metric_res, target,
    #                                     exp_words):
    #     return f'{bool(res)} : {hash_sum} : {threshold_min} : {threshold_max} : {type_prefix} : {model_name} : {metric_res} | {target} =?= {exp_words}\n'
    #
    # @staticmethod
    # def unpack_string_from_prev_results(string):
    #     s = string.strip(' ').split(' | ')[0]
    #     prev_res_str = s.split(' : ')[0]
    #     if prev_res_str == 'True':
    #         prev_res = True
    #     elif prev_res_str == 'False':
    #         prev_res = True
    #     else:
    #         prev_res = 'UNK'
    #     results_1 = s.split(' : ')
    #     hash_sum_str = results_1[1]
    #     # hash_sum = hash_sum_str.strip("'")
    #     hash_sum = hash_sum_str
    #     threshold_min_str = results_1[2]
    #     threshold_max_str = results_1[3]
    #     type_prefix_str = results_1[4]
    #     model_name_str = results_1[5]
    #     metric_res = results_1[6]
    #
    #     s = string.strip(' ').split(' | ')[1]
    #     results_2 = s.split(' =?= ')
    #     target = results_2[0]
    #     exp_words = results_2[1].replace("[", '').replace("'", '').replace(',', '').replace(']', '').replace('\n', '')
    #     exp_words = exp_words.split(' ')
    #     return prev_res, hash_sum, threshold_min_str, threshold_max_str, type_prefix_str, model_name_str, metric_res, target, exp_words


    @staticmethod
    def create_csv_file_for_precomputations(chain_list):
        output_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
        output_dir = os.path.join(output_dir, 'pipeline/chains_generation/')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        pass

        out_file_name = os.path.join(output_dir, "generated_big_csv_file.txt")
        used_hashes = SWUtils.get_used_hashes_from_file(out_file_name)
        # словарь пустой, если файла нет
        if used_hashes is {}:
            csv_headers = ['hash_sum', 'target', 'exp_words']
            SWUtils.write_csv_headers(csv_headers, out_file_name)

        pb = ProgressBar(total=len(chain_list))
        total_added = 0
        total_collisions = 0
        added_hashes = {}

        for element in chain_list:
            pb.print_progress_bar()
            hash_sum = str(Math.get_hash(element))
            if (hash_sum in used_hashes.keys()) or (hash_sum in added_hashes.keys()):
                total_collisions = total_collisions + 1
                continue
            added_hashes[hash_sum] = True

            target = element[0]
            exp_words = element[1:]
            csv_string = [hash_sum, target, exp_words]

            with open(out_file_name, 'a') as fp:
                writer = csv.writer(fp, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(csv_string)
            fp.close()

        print(
            f'Кривой рандом допустил {total_collisions} совпадений, поэтому реально добавили только {total_added} записей')

    # @staticmethod
    # def do_precomputations_by_file_old(unsupported_models: list, do_equity: bool, next_step_count=100000):
    #     f_h_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
    #     out_dir = os.path.join(f_h_dir, 'models_output/')
    #     if not os.path.exists(out_dir):
    #         os.mkdir(out_dir)
    #
    #     do_next_step = True
    #
    #     while do_next_step is True:
    #         models_statuses = {}
    #         models_output_files = {}
    #         for name, model in all_sw_models.items():
    #             if not (name in unsupported_models):
    #                 model_output_file = f'{name}_fh_out.txt'
    #                 model_output_file = os.path.join(out_dir, model_output_file)
    #                 models_output_files[name] = model_output_file
    #                 if os.path.exists(model_output_file):
    #                     models_statuses[name] = Heuristics.file_len(model_output_file)
    #                 else:
    #                     models_statuses[name] = 0
    #
    #         max_precomputations = max(models_statuses.items(), key=operator.itemgetter(1))
    #         min_precomputations = min(models_statuses.items(), key=operator.itemgetter(1))
    #         print(f'Наибольшее количество цепочек ({max_precomputations[1]}) обработала {max_precomputations[0]}')
    #         print(f'Меньше всех цепочек ({min_precomputations[1]}) обработала {min_precomputations[0]}')
    #         max_diff = max_precomputations[1] - min_precomputations[1]
    #
    #         model_name = min_precomputations[0]
    #         if max_diff == 0:
    #             if (max_precomputations[1] == 0) or do_equity is False:
    #                 max_diff = next_step_count
    #                 sw_logger.info(f'Работать, негры! Сейчас {model_name} будет делать ещё {max_diff} цепочек!')
    #             elif do_equity is True:
    #                 do_next_step = False
    #                 sw_logger.info(
    #                     f'Свобода, равенство, браство! Все модели обработали {max_precomputations[1]} цепочек.')
    #                 break
    #
    #         pb = ProgressBar(total=max_diff, epoch_length=(max_diff // 500))
    #
    #         sw_logger.info(f'Сейчас нужно, чтобы {model_name} обработала {max_diff} цепочек.')
    #         model_output_file = models_output_files[model_name]
    #         hashes = {}
    #
    #         i = 0
    #         # читаем хэши уже проверенных цепочек в словарь, чтобы не повторяться в вычислениях
    #         if os.path.exists(model_output_file):
    #             for line in open(model_output_file, 'r'):
    #                 _, has_sum, _, _, _, _, _, _, _ = Heuristics.unpack_string_from_prev_results(line)
    #                 hashes[has_sum] = True
    #
    #         # начинаем читать цепочки из файла
    #         input_file_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
    #         input_file_name = os.path.join(input_file_dir, "generated_big_file.txt")
    #         for line in open(input_file_name, 'r'):
    #
    #             # Если добрали разницу — выходим
    #             if i >= max_diff:
    #                 break
    #
    #             prev_res_str, hash_sum_str, threshold_min_str, threshold_max_str, \
    #             type_prefix_str, model_name_str, \
    #             metric_res, target, exp_words = Heuristics.unpack_string_from_prev_results(line)
    #
    #             # Убеждаемся, что такую мы ещё не считали
    #             if hash_sum_str in hashes.keys():
    #                 continue
    #
    #             i = i + 1
    #
    #             # Проверяем цепочку
    #             target_w = Word(target)
    #             exp_list = [Word(word) for word in exp_words]
    #             chain_validity, metric_res = all_sw_models[model_name].check_explanation_chain_validity(target_w,
    #                                                                                                     exp_list)
    #             pb.print_progress_bar()
    #             # Формируем строку для записи
    #             # определяем имя граничных параметров в зависимости от моделями
    #             if 'word2vec' in model_name:
    #                 t_min_name = 'cosmul_similarity_for_chain_validation_min'
    #                 t_max_name = 'cosmul_similarity_for_chain_validation'
    #             if 'bert' in model_name or 'gpt2' in model_name:
    #                 t_min_name = 'exp_loss_similarity_for_chain_validation_min'
    #                 t_max_name = 'exp_loss_similarity_for_chain_validation_max'
    #             if 'elmo' in model_name:
    #                 t_min_name = 'vec_similarity_for_chain_validation_min'
    #                 t_max_name = 'vec_similarity_for_chain_validation_max'
    #             if 'xlnet' in model_name:
    #                 t_min_name = 'not_implemented_in_sw'
    #                 t_max_name = 'not_implemented_in_sw'
    #
    #             threshold_min = all_sw_models[model_name].params[t_min_name]
    #             threshold_max = all_sw_models[model_name].params[t_max_name]
    #             type_prefix = "fv"
    #
    #             res_str = Heuristics.pack_heuristic_result_to_string(chain_validity, hash_sum_str, threshold_min,
    #                                                                  threshold_max,
    #                                                                  type_prefix, model_name,
    #                                                                  metric_res, target, exp_words)
    #             if chain_validity is True:
    #                 sw_logger.info(f'Найден кандидат! Это строка: {res_str}')
    #             with open(model_output_file, 'a') as fp:
    #                 fp.write(res_str)
    #             fp.close()

    @staticmethod
    def do_precomputations_by_file(unsupported_models: list, do_equity: bool, next_step_count=100000):
        f_h_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
        out_dir = os.path.join(f_h_dir, 'pipeline/improvement/precomputations/')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        do_next_step = True

        while do_next_step is True:
            models_statuses = {}
            models_output_files = {}
            for name, model in all_sw_models.items():
                if not (name in unsupported_models):
                    model_output_file = f'{name}_precs_out.csv'
                    model_output_file = os.path.join(out_dir, model_output_file)
                    models_output_files[name] = model_output_file
                    model_file_len = SWUtils.get_file_len(model_output_file)
                    if model_file_len > 1:
                        models_statuses[name] = model_file_len
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
                    sw_logger.info(
                        f'Свобода, равенство, браство! Все модели обработали {max_precomputations[1]} цепочек.')
                    break

            pb = ProgressBar(total=max_diff)

            sw_logger.info(f'Сейчас нужно, чтобы {model_name} обработала {max_diff} цепочек.')
            model_output_file = models_output_files[model_name]
            hashes = {}

            i = 0
            # # читаем хэши уже проверенных цепочек в словарь, чтобы не повторяться в вычислениях
            # if os.path.exists(model_output_file):
            #     for line in open(model_output_file, 'r'):
            #         _, has_sum, _, _, _, _, _, _, _ = Heuristics.unpack_string_from_prev_results(line)
            #         hashes[has_sum] = True

            # начинаем читать цепочки из файла
            input_file_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
            input_file_name = os.path.join(input_file_dir, "pipeline/improvement/improved_chains.csv")

            csv_header = ['hash_sum', 'generator_name', 'depth_of_generations', 'generation_type',
                          'rankig_model', 'model_rank_score', 'model_decision', 'target', 'exp_words']
            csv_reader = CSVReader(total_operations=next_step_count, input_file_name=input_file_name,
                                   file_to_read_hashes=model_output_file)
            csv_writer = CSVWriter(header=csv_header, total_operations=next_step_count, output_file_name=model_output_file)
            csv_writer.write_csv_header()
            unpacker = Heuristics.unpack_precomputed_chain_result_for_csv_writing

            for line in csv_reader:

                # Если добрали разницу — выходим
                if i >= max_diff:
                    break

                #TODO: работу с хэшами — на уровень ридера, чтобы они повторные туда просто не попадали

                # prev_res_str, hash_sum_str, threshold_min_str, threshold_max_str, \
                # type_prefix_str, model_name_str, \
                # metric_res, target, exp_words = Heuristics.unpack_string_from_prev_results(line)
                #
                # # Убеждаемся, что такую мы ещё не считали
                # if hash_sum_str in hashes.keys():
                #     continue

                i = i + 1

                # TODO: переделать на списки и стинги
                target = line['target']
                exp_words = line['exp_words']
                chain_validity, metric_res = all_sw_models[model_name].check_explanation_chain_validity(target,
                                                                                                        exp_words)
                if chain_validity is True:
                    sw_logger.info(f'Найден кандидат! Это строка: {target} =?= {exp_words}')

                row = (unpacker, [line['model_name'], line['depth_of_generations'], line['generation_type'],
                                  model_name, metric_res, chain_validity, target, exp_words])
                csv_writer.write_csv(row)
                pb.print_progress_bar()

    @staticmethod
    def unpack_ipmproved_chain_result_for_csv_writing(row):
        res = []
        for elements in row:
            for chain in elements:
                model_name = chain[0]
                depth_of_generations = chain[1]
                generation_type = chain[2]
                target = chain[3][0]
                exp_words = chain[3][1:]
                hash_sum = str(Math.get_hash(str(chain[3])))
                res.append([hash_sum, model_name, depth_of_generations, generation_type, target, exp_words])
        return res

    @staticmethod
    def unpack_precomputed_chain_result_for_csv_writing(row):

        res = []
        exp_words = re.sub(r"[^а-яА-Я]+", ' ', row[7]).strip(' ')
        target = row[6]
        full_chain = []
        full_chain.append(target)
        for el in exp_words.split(' '):
            full_chain.append(el)
        hash_sum = str(Math.get_hash(str(full_chain)))
        res = [hash_sum, row[0], row[1], row[2], row[3], row[4], row[5], target, exp_words]
#        line['model_name'], line['depth_of_generations'], line['generation_type'],
#        model_name, metric_res, chain_validity, target, exp_words]

#        ['hash_sum', 'generator_name', 'depth_of_generations', 'generation_type',
#        'rankig_model', 'model_rank_score', 'model_decision', 'target', 'exp_words']

        return res

    @staticmethod
    def improve_chains(chains_file, model_name: str, vocabulary: list, total_improvements: int):
        f_h_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
        output_dir = os.path.join(f_h_dir, 'pipeline/improvement')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir, ' improved_chains.csv')

        voc_vecs = all_sw_models[model_name].calculate_model_sw_vocab_vec(vocabulary)

        csv_reader = CSVReader(total_operations=total_improvements, input_file_name=chains_file)
        csv_headers = ['hash_sum', 'model_name', 'depth_of_generations', 'generation_type', 'target', 'exp_words']
        csv_writer = CSVWriter(total_operations=total_improvements, output_file_name=output_file, header=csv_headers)
        pb = ProgressBar(total=total_improvements)

        unpacker = Heuristics.unpack_ipmproved_chain_result_for_csv_writing
        total_done = 0
        for chain in csv_reader:
            if total_done > total_improvements:
                break
            target = [chain['target']]
            exp_words = chain['exp_words'].replace('[','').replace("'",'').replace(']','').split(', ')
            improved_chains = []
            for i in range(0, 4):
                res = all_sw_models[model_name].improve_chain(target=target, exp_chain=exp_words,
                                                                        vocab_list=vocabulary, word_vecs=voc_vecs,
                                                                        depth=7, type_gen=i)
                improved_chains.append(res)

            row = (unpacker, improved_chains)
            csv_writer.write_csv(row)
            pb.print_progress_bar()
            total_done += 1
        csv_writer.flush()
