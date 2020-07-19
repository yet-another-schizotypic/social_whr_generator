from sw_modelswrapper import all_sw_models, Word, nl_wrapper, Quorum
import scipy
import sw_constants, os
import itertools

import gensim

import fasttext

#ppp = '/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/models/word2vec/araneum_none_fasttextskipgram_300_5_2018/araneum_none_fasttextskipgram_300_5_2018.model'
#model = gensim.models.fasttext.FastTextKeyedVectors.load(ppp)
#d = scipy.spatial.distance.cosine(model.get_vector('кот'), model.get_vector('котейка'))

w1 = Word('когтяра')
#w1_freq = all_sw_models['word2vec_tayga_bow'].get_word_frequency(w1)
#res = nl_wrapper.get_word_frequencies_data(w1)
w2 = Word('котенька')
#w2_freq = all_sw_models['word2vec_tayga_bow'].get_word_frequency(w2)
#res = nl_wrapper.get_word_frequencies_data(w2)

less_frequent_word = nl_wrapper.choose_less_frequent_word(w1, w2)
# res = nl_wrapper.choose_more_frequent_word(w1, w2)


syn_file = os.path.join(sw_constants.SW_SCRIPT_PATH, 'synonyms4test.txt')
out_syn_file = os.path.join(sw_constants.SW_SCRIPT_PATH, 'syns_out.txt')

if os.path.exists(out_syn_file):
    os.remove(out_syn_file)
else:
    print("The file does not exist")

with open(syn_file, "r") as fd:
    lines = fd.read().splitlines()

source_list = []
for line in lines:
    source_list.append(line.strip().lower())

clean_list = list(dict.fromkeys(source_list))

word_list = []
dist = 0
sim = 0
for element in clean_list:
    word_list.append(Word(element.strip().lower()))

for w1, w2 in itertools.combinations(word_list, 2):
    syn_decision = Quorum.check_synonymy(w1, w2)
    #syn_decision, sim = all_sw_models['word2vec_tayga_bow'].check_synonymy_by_relative_cosine_similarity(w1, w2)
    #if syn_decision:
    res_str = 'Слова: {w1} : {w2} : {syn_decision} : {sim}\n'.format(w1=w1.title, w2=w2.title, sim=sim, syn_decision=syn_decision)
    print(res_str)
    with open(out_syn_file, "a") as myfile:
        myfile.write(res_str)





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
# wrong_chains_by_humans = [[['болезнь'], ['игра'], ['дыра'], ['кислород']], ['петух']],\
#                          [[['грудь'], ['галлюцинация'], ['книга'], ['пакет']], ['полицейский']],\
#                          [[['уважение'], ['книга'], ['таблетка'], ['деньги']], ['глаза']],\
#                          [[['глаза'], ['петух'], ['пакет'], ['нож']], ['долг']],\
#                          [[['карандаш'], ['возбуждение'], ['кот'], ['независимость']], ['слон']],\
#                          [[['насилие'], ['ботинок'], ['воля'], ['жидкость']], ['человек']],\
#                          [[['темнота'], ['смелость'], ['френдзона'], ['дыра']], ['закономерность']],\
#                          [[['контекст'], ['надежда'], ['статус'], ['мама']], ['звук']],\
#                          [[['маска'], ['вызов'], ['отказ'], ['треугольник']], ['умиление']],\
#                          [[['счастье'], ['работа'], ['рука'], ['игра']], ['нож']],\
#                          [[['круг'], ['знакомство'], ['навигатор'], ['важность']], ['измена']],\
#                          [[['праздник'], ['время'], ['иерархия'], ['отвержение']], ['книга']],\
#                          [[['камень'], ['любовь'], ['ритуал'], ['новизна']], ['конкуренция']],\
#                          [[['свет'], ['борьба'], ['противоположность'], ['отец']], ['социум']],\
#                          [[['прокол'], ['сон'], ['слон'], ['смерть']], ['равенство']],\
#                          [[['уважение'], ['игра'], ['галлюцинация'], ['контекст']], ['кот']],\
#                          [[['договор'], ['жидкость'], ['слон'], ['дыра']], ['победа']],\
#                          [[['свет'], ['отказ'], ['новизна'], ['травма']], ['камень']],\
#                          [[['слой'], ['карандаш'], ['навигатор'], ['праздник']], ['отвержение']],\
#                          [[['пакет'], ['жидкость'], ['дыра'], ['новизна']], ['понимание']],\
#                          [[['круг'], ['петух'], ['френдзона'], ['победа']], ['мама']],\
#                          [[['помощь'], ['отвращение'], ['слон'], ['иерархия']], ['треугольник']],\
#                          [[['обман'], ['наглость'], ['слой'], ['пакет']], ['любовь']],\
#                          [[['кот'], ['глаза'], ['слон'], ['экзистенциальность']], ['полицейский']],\
#                          [[['сомнение'], ['печаль'], ['радуга'], ['дыра']], ['деньги']],\
#                          [[['пакет'], ['жидкость'], ['дыра'], ['новизна']], ['понимание']],\
#                          [[['круг'], ['петух'], ['френдзона'], ['победа']], ['мама']],\
#                          [[['помощь'], ['отвращение'], ['слон'], ['иерархия']], ['треугольник']],\
#                          [[['обман'], ['наглость'], ['слой'], ['пакет']], ['любовь']],\
#                          [[['кот'], ['глаза'], ['слон'], ['экзистенциальность']], ['полицейский']],\
#                          [[['сомнение'], ['печаль'], ['радуга'], ['дыра']], ['деньги']],\
#                          [[['долг'], ['темнота'], ['измена'], ['статус']], ['юмор']],\
#                          [[['прокол'], ['звук'], ['камень'], ['сон']], ['социум']],\
#                          [[['треугольник'], ['книга'], ['контекст'], ['таблетка']], ['грудь']],\
#                          [[['случайность'], ['насилие'], ['пакет'], ['объект']], ['умиление']]
#
# right_chains_by_humans = [[['работа'], ['интеллект'], ['договор'], ['статус']], ['деньги']],\
#                          [[['рука'], ['признание'], ['симпатия'], ['помощь']], ['благодарность']],\
#                          [[['деньги'], ['работа'], ['воля'], ['борьба']], ['независимость']],\
#                          [[['важность'], ['книга'], ['знакомство'], ['жизнь']], ['интерес']],\
#                          [[['тревога'], ['иллюзия'], ['боязнь'], ['темнота']], ['галлюцинация']],\
#                          [[['отец'], ['соитие'], ['знакомство'], ['любовь']], ['мама']],\
#                          [[['книга'], ['простота'], ['ботинок'], ['нож']], ['пакет']],\
#                          [[['глаза'], ['темнота'], ['отдых'], ['помощь']], ['сон']],\
#                          [[['симпатия'], ['счастье'], ['звук'], ['игра']], ['юмор']],\
#                          [[['работа'], ['деньги'], ['признание'], ['независимость']], ['статус']],\
#                          [[['человек'], ['деньги'], ['темнота'], ['травма']], ['Нож']],\
#                          [[['темнота'], ['травма'], ['пустота'], ['смерть']], ['Нож']],\
#                          [[[' субъект'], ['деньги'], ['провал'], ['разочарование']], ['Обман ']],\
#                          [[['человек'], ['работа'], ['признание'], ['статус']], ['Гордость ']],\
#                          [[['свет'], ['счастье'], ['книга'], ['грудь']], ['Глаза']],\
#                          [[['человек'], ['жизнь'], ['кислород'], ['смерть']], ['Жидкость']],\
#                          [[['противоположность'], ['иерархия'], ['уважение'], ['унижение']], ['Равенство']],\
#                          [[['глаза'], ['болезнь'], ['ритуал'], ['обман']], ['Маска']],\
#                          [[['мать'], ['отец'], ['соитие'], ['счастье']], ['Любовь']],\
#                          [[['фрустрация'], ['потребность'], ['желание'], ['неопределенность']], ['Ожидание']],\
#                          [[['статус'], ['работа'], ['деньги'], ['иерархия']], ['Интеллект']],\
#                          [[['работа'], ['борьба'], ['статус'], ['жизнь']], ['Конкуренция']],\
#                          [[['счастье'], ['дружба'], ['измена'], ['обида']], ['обман']],\
#                          [[['признание'], ['конкуренция'], ['борьба'], ['победа']], ['гордость']],\
#                          [[['голова'], ['сон'], ['темнота'], ['свет']], ['глаза']],\
#                          [[['время'], ['жизнь'], ['болезнь'], ['смерть']], ['жидкость']],\
#                          [[['независимость'], ['человек'], ['дружба'], ['социум']], ['равенство']],\
#                          [[['человек'], ['обман'], ['иллюзия'], ['праздник']], ['маска']],\
#                          [[['человек'], ['счастье'], ['эмпатия'], ['соитие']], ['любовь']],\
#                          [[['неопределенность'], ['тревога'], ['пустота'], ['новизна']], ['ожидание']],\
#                          [[['работа'], ['голова'], ['книга'], ['деньги']], ['интеллект']],\
#                          [[['социум'], ['деньги'], ['потребность'], ['работа']], ['конкуренция']],\
#                          [[['потребность'], ['работа'], ['ожидание'], ['важность']], ['деньги']],\
#                          [[['потребность'], ['помощь'], ['понимание'], ['признание']], ['благодарность']],\
#                          [[['субъект'], ['одиночество'], ['потребность'], ['деньги']], ['независимость']],\
#                          [[['субъект'], ['игра'], ['книга'], ['новизна ']], ['интерес']],\
#                          [[['болезнь '], ['иллюзия'], ['страх'], ['таблетка']], ['галлюцинация']],\
#                          [[['субъект'], ['противоположность '], ['отец'], ['соитие']], ['мама']],\
#                          [[['объект'], ['пустота'], ['дыра'], ['книга']], ['пакет']],\
#                          [[['смерть'], ['время'], ['голова'], ['отдых']], ['сон']],\
#                          [[['человек'], ['книга'], ['социум'], ['звук']], ['юмор']],\
#                          [[['социум'], ['работа'], ['иерархия'], ['деньги']], ['статус']]
#
#
# import os, pickle
#
# fname = 'test'
# extension = '.pickle'
# dirname = '/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/src'
#
#
# i = 0
# for chain in right_chains_by_humans:
#      test_chain = ExplanationChain()
#      test_chain.load_from_string_list(chain)
# #     out_file = open(os.path.join(dirname, (fname + str(i) + extension)), 'wb')
# #     pickle.dump(test_chain, out_file)
# #     out_file.close()
# # #    test_chain.debug_print()
#      t = test_chain.test_quorum_decision(True)
#      i = i + 1
#
# # for i in range(1, 41):
# #     in_file = open(os.path.join(dirname, (fname + str(i) + extension)), 'rb')
# #     loaded_test_chain = pickle.load(in_file)
# #     t = loaded_test_chain.test_quorum_decision(True)
#
# #TODO: нарисовать граф с выразимостью, посмотреть визуально, попробовать определить свойства.
#TODO: посмотреть, обо что и как можно нормализовать вероятность проверки БЕРТа (lognorm,например)