import csv
from os import listdir
from sw_core import SWUtils
import os

def convert_data_from_old_text_tocsv():
    data_dir = '/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/output/heuristics/fileheuristics/models_output'
    for file_name in listdir(data_dir):
        print(f'Конвертируем {file_name}')
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path) or not ('fh_out.txt' in file_name):
            continue
        csv_file_path = os.path.join(data_dir, f'csv_{file_name}.csv'.replace('.txt',''))

        csv_headers = ['hash_sum', 'model_name', 'min_threshold',
                       'max_threshold', 'metric_res', 'model_answer',
                       'type_prefix', 'target', 'exp_words']

        SWUtils.write_csv_headers(csv_headers, csv_file_path)

        for line in open(file_path, 'r'):
            model_answer, hash_sum, threshold_min_str, \
            threshold_max_str, type_prefix_str, model_name_str, \
            metric_res, target, \
            exp_words = SWUtils.unpack_string_from_saved_precomputations_file(line)

            with open(csv_file_path, 'a') as fp:
                writer = csv.writer(fp, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow([hash_sum, model_name_str, threshold_min_str,
                       threshold_max_str, metric_res, model_answer,
                    type_prefix_str, target, exp_words])
            fp.close()


