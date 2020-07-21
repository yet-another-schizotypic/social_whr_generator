from sw_modelswrapper import all_sw_models, Word, nl_wrapper, Quorum
import scipy
from sw_core import sw_logger, ProgressBar
import sw_constants, os
import itertools
import operator



from collections.abc import MutableMapping
# Генератор цепочек
# from sw_graphs import words_graph
# fname = os.path.join(sw_constants.SW_SCRIPT_PATH, 'real_data_by_humans.txt')
# words_graph.initialize_from_file(fname)
# chains = words_graph.get_random_samples_chains(min_len=4, max_len=11, count=1000)
# new_chains_with_titles = []
# for i in range(0, len(chains)):
#     chn = chains[i]
#     tmp_chn = []
#     for element in chn:
#         tmp_chn.append(element.title)
#     new_chains_with_titles.append(tmp_chn)
# print(new_chains_with_titles)
#/Генератор цепочек

#===== TODO: завернуть в функцию, пригодится. Параметры: from, to, step.
filename = os.path.join(sw_constants.SW_SCRIPT_PATH, 'mixed_chains_by_humans.txt')
if os.path.exists(filename):
    with open(filename, "r") as fd:
        lines = fd.read().splitlines()
else:
    raise ('Input file does not exist')

model_name = 'word2vec_tayga_bow'
param_name = 'cosmul_similarity_for_chain_validation'
cat = Word('кот')
cat.get_word_embeddings(model_name)
cycles = 10
pb = ProgressBar(total=cycles, epoch_length=5)
threshold = 0
res_dict = {}

for i in range(0, cycles):
    all_sw_models[model_name].params[param_name] = threshold

    iteration_list = []
    for line in lines:
        desired_res = line.strip(' ').split(' ')[0]
        if desired_res == 'True':
            desired_res = True
        if desired_res == 'False':
            desired_res = False
        target = line.strip(' ').lower().split(' ')[1]
        exp_chain = line.strip(' ').lower().split(' ')[2:]
        exp_words = []
        target = Word(target)
        for exp_word in exp_chain:
            exp_words.append(Word(exp_word))
        res = all_sw_models[model_name].check_explanation_chain_validity(target, exp_words)
        if res == desired_res:
            iteration_list.append(True)
        else:
            iteration_list.append(False)

    res_dict[threshold] = sum(iteration_list)
    threshold = threshold + 1 / cycles
    pb.print_progress_bar()

max_index = max(res_dict.items(), key=operator.itemgetter(1))[0]
max_value = res_dict[max_index]
print(f'Наибольшее количество совпадений: {max_value} при значении {max_index}')
# TODO: Прогнать словари через stopwords, чтобы там всяких предлогов не осталось
#TODO: Хэшировать протестированные кворумом цепочки, чтобы избегать последующего ре-тестирования на той же версии кворума

# w = Word('картошка')
# emb = w.get_word_embeddings('word2vec_tayga_bow')
# print(emb)
#
# emb = w.get_word_embeddings('word2vec_tayga_bow')
#
# emb = w.get_word_embeddings('gpt2_russian_finetuning')
#
# test = all_sw_models['word2vec_tayga_bow'].get_embeddings('кот')
# for key, value in all_sw_models.items():
#
#     cat_emb = value.get_embeddings('кот')
#     kitten_emb = value.get_embeddings('котёнок')
#     train_emb = value.get_embeddings('поезд')
#
#     print(str(key))
#     print('Кот - котёнок: расстояние {dist}'.format(dist=scipy.spatial.distance.cosine(cat_emb, kitten_emb)))
#     print('Поезд - котёнок: расстояние {dist}'.format(dist=scipy.spatial.distance.cosine(train_emb, kitten_emb)))
#


#
# #TODO: нарисовать граф с выразимостью, посмотреть визуально, попробовать определить свойства.
# TODO: посмотреть, обо что и как можно нормализовать вероятность проверки БЕРТа (lognorm,например)
# TODO: в прогресс-бар впихнуть на первую итерацию реальное время начала, на последнюю — фактическое время выполнения