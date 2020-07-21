from datetime import datetime, timedelta
# from sw_core import StopTimer
# #st = StopTimer(duration="00:45:02", tick="00:00:5")
# st = StopTimer(end_time="19:21:02", tick="00:00:5")
# while True:
#      if st.check_time_has_gone() is True:
#          break
#      pass

from sw_graphs import WordGraph
import sw_constants
from sw_heuristics import Heuristics
import os


fname = os.path.join(sw_constants.SW_SCRIPT_PATH, 'real_data_by_humans.txt')
# words_graph.initialize_from_file(fname)
wg = WordGraph()
wg.initialize_from_file(fname)
words_list = wg.get_all_words_from_dict()
heur = Heuristics()
heur.find_canditates_chains_by_simple_cosmul(words_list, 'word2vec_tayga_bow')