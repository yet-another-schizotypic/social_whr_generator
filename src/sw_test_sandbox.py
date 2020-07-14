# TODO:
# Список хороших объяснений (где-то был) list
# Список плохих цепочек (в Телеге)
# Написать валидаторы по всем моделям и проверить, какой лучше


from sw_core import *

word1 = Word('хуй')
word2 = Word('пизда')
word3 = Word('жопа')
word4 = Word('блядь')

wrong_chains_by_humans = [[['болезнь'], ['игра'], ['дыра'], ['кислород']], ['петух']],\
                         [[['грудь'], ['галлюцинация'], ['книга'], ['пакет']], ['полицейский']],\
                         [[['уважение'], ['книга'], ['таблетка'], ['деньги']], ['глаза']],\
                         [[['глаза'], ['петух'], ['пакет'], ['нож']], ['долг']],\
                         [[['карандаш'], ['возбуждение'], ['кот'], ['независимость']], ['слон']],\
                         [[['насилие'], ['ботинок'], ['воля'], ['жидкость']], ['человек']],\
                         [[['темнота'], ['смелость'], ['френдзона'], ['дыра']], ['закономерность']],\
                         [[['контекст'], ['надежда'], ['статус'], ['мама']], ['звук']],\
                         [[['маска'], ['вызов'], ['отказ'], ['треугольник']], ['умиление']],\
                         [[['счастье'], ['работа'], ['рука'], ['игра']], ['нож']],\
                         [[['круг'], ['знакомство'], ['навигатор'], ['важность']], ['измена']],\
                         [[['праздник'], ['время'], ['иерархия'], ['отвержение']], ['книга']],\
                         [[['камень'], ['любовь'], ['ритуал'], ['новизна']], ['конкуренция']],\
                         [[['свет'], ['борьба'], ['противоположность'], ['отец']], ['социум']],\
                         [[['прокол'], ['сон'], ['слон'], ['смерть']], ['равенство']],\
                         [[['уважение'], ['игра'], ['галлюцинация'], ['контекст']], ['кот']],\
                         [[['договор'], ['жидкость'], ['слон'], ['дыра']], ['победа']],\
                         [[['свет'], ['отказ'], ['новизна'], ['травма']], ['камень']],\
                         [[['слой'], ['карандаш'], ['навигатор'], ['праздник']], ['отвержение']],\
                         [[['пакет'], ['жидкость'], ['дыра'], ['новизна']], ['понимание']],\
                         [[['круг'], ['петух'], ['френдзона'], ['победа']], ['мама']],\
                         [[['помощь'], ['отвращение'], ['слон'], ['иерархия']], ['треугольник']],\
                         [[['обман'], ['наглость'], ['слой'], ['пакет']], ['любовь']],\
                         [[['кот'], ['глаза'], ['слон'], ['экзистенциальность']], ['полицейский']],\
                         [[['сомнение'], ['печаль'], ['радуга'], ['дыра']], ['деньги']],\
                         [[['пакет'], ['жидкость'], ['дыра'], ['новизна']], ['понимание']],\
                         [[['круг'], ['петух'], ['френдзона'], ['победа']], ['мама']],\
                         [[['помощь'], ['отвращение'], ['слон'], ['иерархия']], ['треугольник']],\
                         [[['обман'], ['наглость'], ['слой'], ['пакет']], ['любовь']],\
                         [[['кот'], ['глаза'], ['слон'], ['экзистенциальность']], ['полицейский']],\
                         [[['сомнение'], ['печаль'], ['радуга'], ['дыра']], ['деньги']],\
                         [[['долг'], ['темнота'], ['измена'], ['статус']], ['юмор']],\
                         [[['прокол'], ['звук'], ['камень'], ['сон']], ['социум']],\
                         [[['треугольник'], ['книга'], ['контекст'], ['таблетка']], ['грудь']],\
                         [[['случайность'], ['насилие'], ['пакет'], ['объект']], ['умиление']]

right_chains_by_humans = [[['работа'], ['интеллект'], ['договор'], ['статус']], ['деньги']],\
                         [[['рука'], ['признание'], ['симпатия'], ['помощь']], ['благодарность']],\
                         [[['деньги'], ['работа'], ['воля'], ['борьба']], ['независимость']],\
                         [[['важность'], ['книга'], ['знакомство'], ['жизнь']], ['интерес']],\
                         [[['тревога'], ['иллюзия'], ['боязнь'], ['темнота']], ['галлюцинация']],\
                         [[['отец'], ['соитие'], ['знакомство'], ['любовь']], ['мама']],\
                         [[['книга'], ['простота'], ['ботинок'], ['нож']], ['пакет']],\
                         [[['глаза'], ['темнота'], ['отдых'], ['помощь']], ['сон']],\
                         [[['симпатия'], ['счастье'], ['звук'], ['игра']], ['юмор']],\
                         [[['работа'], ['деньги'], ['признание'], ['независимость']], ['статус']],\
                         [[['человек'], ['деньги'], ['темнота'], ['травма']], ['Нож']],\
                         [[['темнота'], ['травма'], ['пустота'], ['смерть']], ['Нож']],\
                         [[[' субъект'], ['деньги'], ['провал'], ['разочарование']], ['Обман ']],\
                         [[['человек'], ['работа'], ['признание'], ['статус']], ['Гордость ']],\
                         [[['свет'], ['счастье'], ['книга'], ['грудь']], ['Глаза']],\
                         [[['человек'], ['жизнь'], ['кислород'], ['смерть']], ['Жидкость']],\
                         [[['противоположность'], ['иерархия'], ['уважение'], ['унижение']], ['Равенство']],\
                         [[['глаза'], ['болезнь'], ['ритуал'], ['обман']], ['Маска']],\
                         [[['мать'], ['отец'], ['соитие'], ['счастье']], ['Любовь']],\
                         [[['фрустрация'], ['потребность'], ['желание'], ['неопределенность']], ['Ожидание']],\
                         [[['статус'], ['работа'], ['деньги'], ['иерархия']], ['Интеллект']],\
                         [[['работа'], ['борьба'], ['статус'], ['жизнь']], ['Конкуренция']],\
                         [[['счастье'], ['дружба'], ['измена'], ['обида']], ['обман']],\
                         [[['признание'], ['конкуренция'], ['борьба'], ['победа']], ['гордость']],\
                         [[['голова'], ['сон'], ['темнота'], ['свет']], ['глаза']],\
                         [[['время'], ['жизнь'], ['болезнь'], ['смерть']], ['жидкость']],\
                         [[['независимость'], ['человек'], ['дружба'], ['социум']], ['равенство']],\
                         [[['человек'], ['обман'], ['иллюзия'], ['праздник']], ['маска']],\
                         [[['человек'], ['счастье'], ['эмпатия'], ['соитие']], ['любовь']],\
                         [[['неопределенность'], ['тревога'], ['пустота'], ['новизна']], ['ожидание']],\
                         [[['работа'], ['голова'], ['книга'], ['деньги']], ['интеллект']],\
                         [[['социум'], ['деньги'], ['потребность'], ['работа']], ['конкуренция']],\
                         [[['потребность'], ['работа'], ['ожидание'], ['важность']], ['деньги']],\
                         [[['потребность'], ['помощь'], ['понимание'], ['признание']], ['благодарность']],\
                         [[['субъект'], ['одиночество'], ['потребность'], ['деньги']], ['независимость']],\
                         [[['субъект'], ['игра'], ['книга'], ['новизна ']], ['интерес']],\
                         [[['болезнь '], ['иллюзия'], ['страх'], ['таблетка']], ['галлюцинация']],\
                         [[['субъект'], ['противоположность '], ['отец'], ['соитие']], ['мама']],\
                         [[['объект'], ['пустота'], ['дыра'], ['книга']], ['пакет']],\
                         [[['смерть'], ['время'], ['голова'], ['отдых']], ['сон']],\
                         [[['человек'], ['книга'], ['социум'], ['звук']], ['юмор']],\
                         [[['социум'], ['работа'], ['иерархия'], ['деньги']], ['статус']]


import os, pickle

fname = 'test'
extension = '.pickle'
dirname = '/Users/yet-another-schizotypic/Documents/__Develop/Социоблядь/social_whr_generator/src'


i = 0
# for chain in right_chains_by_humans:
#     test_chain = ExplanationChain()
#     test_chain.load_from_string_list(chain)
#     out_file = open(os.path.join(dirname, (fname + str(i) + extension)), 'wb')
#     pickle.dump(test_chain, out_file)
#     out_file.close()
# #    test_chain.debug_print()
#     t = test_chain.test_quorum_decision(True)
#     i = i + 1

for i in range(1, 41):
    in_file = open(os.path.join(dirname, (fname + str(i) + extension)), 'rb')
    loaded_test_chain = pickle.load(in_file)
    t = loaded_test_chain.test_quorum_decision(True)

#TODO: нарисовать граф с выразимостью, посмотреть визуально, попробовать определить свойства.
#TODO: посмотреть, обо что и как можно нормализовать вероятность проверки БЕРТа (lognorm,например)
print()