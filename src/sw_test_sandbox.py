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


def unpack_csv_string(row):
    res = [[row[0], row[1]]]
    return res
def test_csv_buffers():
    from sw_core import CSVReader, CSVWriter
    import sw_constants
    file_name = os.path.join(sw_constants.SW_SCRIPT_PATH, 'excel_csv_test.csv')
    output_file = os.path.join(sw_constants.SW_SCRIPT_PATH, 'excel_csv_test_output.csv')
    csv_writer = CSVWriter(total_operations=17, header=['digits', 'letters'], output_file_name=output_file)
    csv_reader = CSVReader(input_file_name=file_name, total_operations=13)
    unpacker = unpack_csv_string
    for row in csv_reader:
        res = (unpacker, [row['digits'], row['letters']])
        csv_writer.write_csv(res)

def test_magnitude():
    import pymagnitude
    from pymagnitude import converter
    input_dir = "/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/models/ELMo/elmo_src"
    output_dir = "/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/models/ELMo/elmo_magnitude"
    converter.convert(input_file_path= input_dir, output_file_path=output_dir)

test_magnitude()