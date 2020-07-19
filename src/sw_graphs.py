import scipy
from networkx import DiGraph
import matplotlib.pyplot as plt
import itertools
import collections


class WordGraph(collections.UserDict, DiGraph):
    def __init__(self):

        pass


wg = WordGraph()

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
