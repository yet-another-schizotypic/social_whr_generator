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
        csv_file_path = os.path.join(data_dir, f'csv_{file_name}.csv'.replace('.txt', ''))

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


# def test_magnitude():
#     import pymagnitude
#     from pymagnitude import converter
#     input_dir = "/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/models/ELMo/elmo_src"
#     output_dir = "/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/models/ELMo/elmo_magnitude"
#     converter.convert(input_file_path= input_dir, output_file_path=output_dir)
#
# test_magnitude()

def get_most_similar_cosmul_from_vocab(model_name):
    from sw_modelswrapper import all_sw_models
    from sw_core import config_parser
    import sw_constants
    from gensim.models.keyedvectors import Vocab

    f_h_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
    input_dir = os.path.join(f_h_dir, 'pipeline')

    vocab_file_name = os.path.join(sw_constants.SW_SCRIPT_DATA_PATH, 'united_dict.txt')
    vocabulary = SWUtils.read_vocab_without_duplicates(vocab_file_name, check_synonymy=False)
    all_sw_models[model_name].check_init_model_state()
    voc_vecs = all_sw_models[model_name].calculate_model_sw_vocab_vec(vocabulary)

    voc_dict = {}
    i = 0
    for element in vocabulary:
        v = Vocab(index=i)
        voc_dict[element] = v
        i += 1

    # Дальше — просто тест на человечекских цепочках
    input_file = os.path.join(sw_constants.SW_SCRIPT_PATH, 'mixed_chains_by_humans.txt')
    for line in open(input_file, 'r'):
        human_target = line.lower().split('=')[0]
        exp_words = line.lower().replace('+', ' ').split('=')[1].strip('\n').split(' ')
        positive = exp_words
        machine_suggestions_cosmul = all_sw_models[model_name].get_most_similar_cosmul_from_vocab(
            entities_list=vocabulary,
            vocab_dict=voc_dict,
            wvecs=voc_vecs, positive=positive)
        machine_suggestions = all_sw_models[model_name].get_most_similar_from_vocab(entities_list=vocabulary,
                                                                                    wvecs=voc_vecs, positive=positive)

        print(
            f'Цепочка: «{line.lower().split("=")[1:]}». Ответ человека: «{human_target}». Догадки машины: «{machine_suggestions_cosmul}»')


def combination_unpacker(row):
    return [row[0], row[1:]]

from pymagnitude import *

def get_combinations():
    from sw_core import config_parser, ProgressBar, CSVWriter
    import sw_constants
    import itertools
    from scipy import special

    f_h_dir = config_parser.config['sw_dirs']['file_heuristics_dir']
    vocab_file_name = os.path.join(sw_constants.SW_SCRIPT_DATA_PATH, 'united_dict.txt')
    output_file = os.path.join(sw_constants.SW_SCRIPT_DATA_PATH, 'combinations.txt')
    last_iter = os.path.join(sw_constants.SW_SCRIPT_DATA_PATH, 'last_iter.txt')
    vocabulary = SWUtils.read_vocab_without_duplicates(vocab_file_name, check_synonymy=False)
    comb_count = special.comb(len(vocabulary), 6)
    pb = ProgressBar(total=comb_count)
    m_dir = '/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/models/pymagnitude/pymagnitude_tayga_lemmas_ELMo/'
    mag_file = os.path.join(m_dir, 'model.magnitude')
    vectors = Magnitude(mag_file)

    fp = open(output_file, 'w')
    i = 0
    j=0
    for combination in itertools.combinations(vocabulary, 6):
        target = list([combination[0]])
        exp_words = list(combination[1:])
        dist = vectors.distance(exp_words, target)
        if (dist[0]>=0) and (dist[0]<=61.916):
            print(str(combination), file=fp)
            print(str(combination), i)
        pb.print_progress_bar()
        i+=1
        j+=1
        if j>100000:
            with open(last_iter, 'w') as fp:
                fp.write(str(i))
            fp.close()
            print(f'Сделали {i} итераций')




def pymagnitude_test():

    m_dir = '/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/models/pymagnitude/pymagnitude_tayga_lemmas_ELMo/'
    mag_file = os.path.join(m_dir, 'model.magnitude')
    vectors = Magnitude(mag_file)
    res = vectors.query(["I", "read", "a", "book"])
    vectors.most_similar_cosmul(positive=["woman", "king"], negative=["man"])
    print(vectors.similarity('нож', ["подарок", "лужа", 'крехер']))

run = get_combinations()


