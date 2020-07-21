from random import randint

import scipy
from networkx import DiGraph
import matplotlib.pyplot as plt
import itertools
import collections
from sw_core import sw_logger, ProgressBar
from sw_modelswrapper import Word, Quorum, nl_wrapper
from collections.abc import MutableMapping
import os


class WordGraph(MutableMapping, DiGraph):

    def __init__(self, incoming_graph_data=None, *args, **attr):
        self.store = dict()
        self.update(dict(*args, **attr))  # use the free update to set keys
        super(DiGraph, self).__init__(incoming_graph_data=None, **attr)

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key

    def get_all_words_from_dict(self):
        res_list = []
        for element in self.store.values():
            res_list.append(element)
        return res_list

    def get_random_samples_chains(self, min_len, max_len, count):
        chains_list = []
        for i in range(0, count):
            chain_len = randint(min_len, max_len)
            used_indexes = []
            flg = False
            words_in_chain = []
            rand_index = -1
            for j in range(0, chain_len):
                while flg is False:
                    rand_index = randint(0, len(self) - 1)
                    if not (rand_index in used_indexes):
                        flg = True
                        used_indexes.append(rand_index)

                flg = False
                word = list(self.store.values())[rand_index]
                words_in_chain.append(word)

            chains_list.append(words_in_chain)
        return chains_list


    def initialize_from_file(self, filename, check_synonymy = False):
        if os.path.exists(filename):
            with open(filename, "r") as fd:
                lines = fd.read().splitlines()
        else:
            raise ('Input file does not exist')

        source_list = []
        for line in lines:
            source_list.append(line.strip().lower())

        clean_list = list(dict.fromkeys(source_list))
        diff_count = len(source_list) - len(clean_list)
        if diff_count != 0:
            sw_logger.info("Найдено {dc} буквальных дубликатов, дубликаты удалены.".format(dc=diff_count))

        word_list = []
        for element in clean_list:
            word_list.append(Word(element.strip().lower()))

        if check_synonymy is True:
            total_comb_count = scipy.special.comb(len(word_list), 2)
            pb = ProgressBar(total=total_comb_count, epoch_length=2000)
            sw_logger.info('Начинаем проверку на синонимичность')
            for w1, w2 in itertools.combinations(word_list, 2):
                syn_decision = Quorum.check_synonymy(w1, w2)
                if syn_decision is True:
                    lfw = nl_wrapper.choose_less_frequent_word(w1, w2)
                    if lfw in word_list:
                        word_list.remove(lfw)
                        sw_logger.info(
                            'Слова «{w1}» и «{w2}» — синонимы (в понимании модели), слово «{lfw}» встречается реже, '
                            'удаляем его.'.format(w1=w1.title, w2=w2.title, lfw=lfw.title))
                pb.print_progress_bar()
            sw_logger.info('Проверка на синонимичность завершена.')

        for element in word_list:
            element.suggested_by = 'human'
            self[element.title] = element
            super(DiGraph, self).add_node(element.title, suggested_by=element.suggested_by,
                                          tested_as_target=element.tested_as_target,
                                          tested_in_explanation_chains=element.tested_in_explanation_chains,
                                          succeeded_as_target=element.succeeded_as_target,
                                          succeeded_in_explanation_chains=element.succeeded_in_explanation_chains)


words_graph = WordGraph()

# class WordSet(nx.classes.graph.Graph):
#     def load_from_word_list(self, word_list: list):
#         for word in word_list:
#             word_object = Word(word)
#             self.add_node(word_object.title, word2vec_title=word_object.word2vec_title,
#                           word2vec_embeddings=word_object.__word2vec_embeddings,
#                           bert_embeddings=word_object.bert_embeddings, elmo_embeddings=word_object.elmo_embeddings)
#
#     def draw_graph(self):
#         nx.draw(self, with_labels=True)
#         plt.show()
#
#     def __get_all_nodes_words(self):
#         return list(self.nodes)
#
#     def __naive_remove_synonyms_nodes(self, embeddings_type=sw_constants.BERT):
#         typedef = sw_constants.ModelSpecificSettings(embeddings_type)
#         data_type = typedef.data_type_for_nx_graph
#         min_cosine_neighbourhood_distance = typedef.min_cosine_neighbourhood_distance
#
#         deleted_items = []
#         print('Ищем и удаляем «лишние» слова...')
#         for a, b in itertools.combinations(self.nodes(data=data_type), 2):
#             print('Смотрим пару', a[0], b[0])
#             dist = math.get_distance(a[1], b[1])
#             if dist < min_cosine_neighbourhood_distance:
#                 if not (b[0] in deleted_items):
#                     self.remove_node(b[0])
#                     print('Удалён узел: «', b[0], '» из пары «', a[0], '» — «', b[0], '»', dist)
#                     deleted_items.append(b[0])
#             else:
#                 print('Оставляем', a[0], b[0], dist)
#
#     def naive_improve_connectivity_step(self, embeddings_type=sw_constants.BERT):
#
#         self.__naive_remove_synonyms_nodes(embeddings_type)
#
#         typedef = sw_constants.ModelSpecificSettings(embeddings_type)
#         data_type = typedef.data_type_for_nx_graph
#         clear_embeddings_vec = typedef.clear_embeddings_vec
#         max_cosine_neighbourhood_distance = typedef.max_cosine_neighbourhood_distance
#
#         print('Добавляем новые слова')
#         # Второй цикл — после прохождения первого состав мог измениться
#         too_long_pairs = []
#         for a, b in itertools.combinations(self.nodes(data=data_type), 2):
#             dist = math.get_distance(a[1], b[1])
#             if dist > max_cosine_neighbourhood_distance:
#                 too_long_pairs.append([a[0], b[0]])
#                 print(a[0], b[0], dist)
#
#         # Собрали все слишком далёкие вершины, ищем варианты расширения набора
#         # TODO: не всегда тут должен быть BERT, поставить условие на embeddings_type
#         all_words = self.__get_all_nodes_words()
#         candidates = NLWrapper.generate_node_candidates_BERT(pairs=too_long_pairs,
#                                                                  complete_word_set=all_words)
#
#         # TODO: Наивное сравнение на циклах. В word2vec можно mostsimilartogiven
#
#         new_words = []
#         for candidate in candidates:
#             new_word = Word(candidate)
#             new_words.append(new_word)
#
#         # TODO: сравнение расстояния может быть не только по word2vec
#         vec = clear_embeddings_vec.copy()
#         for node in self.nodes(data=data_type):
#             vec = math.sum_vectors(vec, node[1])
#
#         distances = {}
#         for new_word in new_words:
#             if embeddings_type == sw_constants.WORD2VEC:
#                 dist = math.get_distance(vec, new_word.__word2vec_embeddings)
#             if embeddings_type == sw_constants.BERT:
#                 dist = math.get_distance(vec, new_word.bert_embeddings)
#             distances.update({dist: new_word.title})
#
#         distances = collections.OrderedDict(sorted(distances.items()))
#
#         print(distances)
#
# word_list = ('усы', 'кошка', 'лапы', 'хвост', 'мяуканье', 'кот', 'кошечка', 'френдзона')
# word_set = WordSet()
# word_set.load_from_word_list(word_list)
# word_set.naive_improve_connectivity_step()
# # word_set.draw_graph()
