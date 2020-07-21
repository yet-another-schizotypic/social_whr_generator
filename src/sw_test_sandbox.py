from datetime import datetime, timedelta
from sw_core import StopTimer
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
wg = WordGraph()
wg.initialize_from_file(fname)
words_list = wg.get_all_words_from_dict()
heur = Heuristics()
stop_timer = StopTimer(end_time="22:15:00", tick="00:05:00")
heur.find_canditates_chains_by_processing_elements(words_list=words_list, model_name='word2vec_tayga_bow',
                                                   stop_timer=stop_timer)