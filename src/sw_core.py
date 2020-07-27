# TODO:
# 1. Вынести сюда Math и Word
# 2. Стащить из Permutations генератор псевдо-рандомных выборок
# 3 Сделать очень разные BERT'ы
# 4. В классе графа использовать в атрибутах константы из констант
# 5. Сделать много разныъ бертов, декорировать
# 6. Посмотреть в softmax и около для проверки валидности цепочек
# 7. Класс Word_List с синонимами и произвольной выборкой комбинаций
# 8. Класс chain с проверками валидности
# 9. Добавить разных БЕРТов, отврефакторить имеющийся
# 10. Ембеддинги DistilBert, Elmo
import itertools
from random import randint

import typing

import sw_constants
import logging
import os, pathlib, json

sw_logger = logging.getLogger('socialwhore_loger')
sw_logger.setLevel(sw_constants.LOGLEVEL)
sw_format = logging.Formatter('%(asctime)s - %(message)s')
sw_handler = logging.StreamHandler()
sw_handler.setFormatter(sw_format)
sw_logger.addHandler(sw_handler)

import numpy as np

import scipy, hashlib, base64
from datetime import datetime, timedelta, time
import csv


class SWConfigParser:
    def __init__(self, config_file_name=sw_constants.SW_CONFIG_FILE_NAME):
        __project_path__ = pathlib.Path(__file__).parent.absolute()
        config_file = os.path.join(__project_path__, config_file_name)
        with open(config_file, 'r') as fp:
            data = json.load(fp)
        fp.close()
        n_m_flag = False
        sw_supported_models = {}
        for key, value in data.items():
            n_m_flag = False
            if isinstance(value, dict):
                for n, v in value.items():
                    if isinstance(v, list):
                        i = 0
                        for element in v:
                            if os.path.isdir(str('.' + element)) or os.path.isfile(str('.' + element)):
                                v[i] = os.path.join(__project_path__, str('.' + v[i])).replace('/./', '/')
                                i = i + 1
                        sw_supported_models = {**sw_supported_models, **value}
                    elif isinstance(v, dict):
                        for ke, vl in v.items():
                            if os.path.isdir(str('.' + str(vl))) or os.path.isfile(str('.' + str(vl))):
                                v[ke] = os.path.join(__project_path__, str('.' + v[ke])).replace('/./', '/')
                                n_m_flag = True
                        continue

                    elif os.path.isdir(str('.' + v)) or os.path.isfile(str('.' + v)):
                        value[n] = os.path.join(__project_path__, str('.' + v)).replace('/./', '/')

                if n_m_flag is False:
                    sw_supported_models = {**sw_supported_models, **value}

        all_models = {'sw_supported_models': sw_supported_models}
        data = {**data, **all_models}
        self.config = data.copy()


config_parser = SWConfigParser()


class SWUtils:

    # TODO переделать на CSV, удалить аналог из Heuristics
    @staticmethod
    def unpack_string_from_saved_precomputations_file(string):
        s = string.strip(' ').split(' | ')[0]
        prev_res_str = s.split(' : ')[0]
        if prev_res_str == 'True':
            prev_res = True
        elif prev_res_str == 'False':
            prev_res = True
        else:
            prev_res = 'UNK'
        results_1 = s.split(' : ')
        hash_sum_str = results_1[1]
        # hash_sum = hash_sum_str.strip("'")
        hash_sum = hash_sum_str
        threshold_min_str = results_1[2]
        threshold_max_str = results_1[3]
        type_prefix_str = results_1[4]
        model_name_str = results_1[5]
        metric_res = results_1[6]

        s = string.strip(' ').split(' | ')[1]
        results_2 = s.split(' =?= ')
        target = results_2[0]
        exp_words = results_2[1].replace("[", '').replace("'", '').replace(',', '').replace(']', '').replace('\n', '')
        exp_words = exp_words.split(' ')
        return prev_res, hash_sum, threshold_min_str, threshold_max_str, type_prefix_str, model_name_str, metric_res, target, exp_words

    @staticmethod
    def get_file_len(file_name):
        if (not os.path.exists(file_name)) or (file_name is None):
            return 0
        with open(file_name) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    @staticmethod
    def generate_quasi_random_samples_from_string_list(string_list: list, min_len, max_len, count):
        pb = ProgressBar(total=count)
        chains_list = []
        for i in range(0, count):
            chain_len = randint(min_len, max_len)
            used_indexes = []
            flg = False
            words_in_chain = []
            rand_index = -1
            for j in range(0, chain_len):
                while flg is False:
                    rand_index = randint(0, len(string_list) - 1)
                    if not (rand_index in used_indexes):
                        flg = True
                        used_indexes.append(rand_index)
                flg = False
                word = string_list[rand_index]
                words_in_chain.append(word)
            chains_list.append(words_in_chain)
            pb.print_progress_bar()
        return chains_list

    @staticmethod
    def file_has_more_than_one_line(file_path):
        if file_path is None:
            return False
        if not (os.path.exists(file_path)):
            return False
        i = 0
        for line in open(file_path, 'r'):
            i += 1
            if i > 1:
                return True

    @staticmethod
    def read_vocab_without_duplicates(file_name, check_synonymy: bool):
        if os.path.exists(file_name):
            with open(file_name, "r") as fd:
                lines = fd.read().splitlines()
        else:
            raise ValueError('Input file does not exist')

        source_list = []
        for line in lines:
            source_list.append(line.strip().lower())

        clean_list = list(dict.fromkeys(source_list))
        diff_count = len(source_list) - len(clean_list)
        if diff_count != 0:
            sw_logger.info("Найдено {dc} буквальных дубликатов, дубликаты удалены.".format(dc=diff_count))

        # TODO: этот кусок не работает, и неизвестно, понадобится ли. К удалению.
        if check_synonymy is True:
            raise ValueError('Not implemented')
        #     total_comb_count = scipy.special.comb(len(clean_list), 2)
        #     pb = ProgressBar(total=total_comb_count)
        #     sw_logger.info('Начинаем проверку на синонимичность')
        #     for w1, w2 in itertools.combinations(clean_list, 2):
        #         syn_decision = Quorum.check_synonymy(w1, w2)
        #         if syn_decision is True:
        #             lfw = nl_wrapper.choose_less_frequent_word(w1, w2)
        #             if lfw in word_list:
        #                 word_list.remove(lfw)
        #                 sw_logger.info(
        #                     'Слова «{w1}» и «{w2}» — синонимы (в понимании модели), слово «{lfw}» встречается реже, '
        #                     'удаляем его.'.format(w1=w1.title, w2=w2.title, lfw=lfw.title))
        #         pb.print_progress_bar()
        #     sw_logger.info('Проверка на синонимичность завершена.')
        return clean_list

    @staticmethod
    def unpack_word_objects_list_to_string_list(word_object_list):
        return [word.title for word in word_object_list]


# Буффер CSV
class CSVBuffer:
    def __init__(self, total_operations: int, file_name: str, header: list):
        if total_operations is None:
            self.buffer_size = 1000
        if total_operations >= 1000000:
            self.buffer_size = 50000
        elif total_operations >= 500000:
            self.buffer_size = 15000
        elif total_operations >= 100000:
            self.buffer_size = 10000
        elif total_operations >= 10000:
            self.buffer_size = 2000
        elif total_operations >= 1000:
            self.buffer_size = 200
        elif total_operations >= 100:
            self.buffer_size = 10
        else:
            self.buffer_size = 5

        self.buffer_list = []
        self.file_to_read_hashes_from = file_name
        self.header = header
        self.used_hashes = {}
        if not (self.file_to_read_hashes_from is None):
            if SWUtils.file_has_more_than_one_line(self.file_to_read_hashes_from):
                with open(self.file_to_read_hashes_from) as fp:
                    reader = csv.DictReader(fp)
                    for line in reader:
                        self.used_hashes[line['hash_sum']] = True


class CSVReader(CSVBuffer):
    def __init__(self, total_operations: int, input_file_name: str, file_to_read_hashes=None):
        super().__init__(total_operations, file_to_read_hashes, None)
        self.file_to_read = input_file_name
        self.last_read_position = 0
        self.__csv_reader__ = None
        if SWUtils.file_has_more_than_one_line(input_file_name):
            self.__file_object__ = open(self.file_to_read, 'r')
            self.__file_descriptor__ = self.__file_object__.fileno()
            self.__perm_file_object__ = fp = open(self.__file_descriptor__, 'r', closefd=False)
            self.__csv_reader__ = csv.DictReader(fp)
            self.header = next(self.__csv_reader__)


    def __read_csv_file_to_inner_buffer__(self, include_header=False, start_position=0):
        self.buffer_list = []
        i = 0

        for line in self.__csv_reader__:
            self.buffer_list.append(line)
            i += 1
            if i >= self.buffer_size:
                break

    # def read_line_from_csv_file(self):
    #     if len(self.buffer_list) == 0:
    #         self.__read_csv_file_to_inner_buffer__()
    #
    #     while True:
    #         for element in self.buffer_list:
    #             yield element
    #         self.buffer_list = []
    #         self.__read_csv_file_to_inner_buffer__()
    #         #TODO: Если буффер пустой, выходим
    #         #TODO: протестировать буфферы на Экселе на предмет не теряют ли они данные
    #         #Это всё вообще имеет смысл, если получится найти в Питоне fseek

    def __iter__(self):
        if len(self.buffer_list) == 0:
            self.__read_csv_file_to_inner_buffer__()

        while True:
            for element in self.buffer_list:
                yield element
            self.buffer_list = []
            self.__read_csv_file_to_inner_buffer__()

    def __del__(self):
        self.__perm_file_object__.close()


class CSVWriter(CSVBuffer):
    def __init__(self, total_operations: int, output_file_name: str, header: list):
        super().__init__(total_operations, output_file_name, header)

    def flush(self):
        self.write_csv_header(print_warning=False)
        # print(f'Сбрасываю буффер из {len(self.buffer_list)} записей в файл {self.file_to_read_hashes_from}')
        with open(self.file_to_read_hashes_from, 'a') as fp:
            writer = csv.writer(fp, quoting=csv.QUOTE_NONNUMERIC)
            for element in self.buffer_list:
                writer.writerow(element)
        fp.close()
        self.buffer_list = []


    def write_csv(self, row):
        assert row[0], typing.Callable
        unpacker = row[0]
        lines = unpacker(row[1])
        for line in lines:
            if line[0] in self.used_hashes.keys():
                continue
            self.used_hashes[line[0]] = True
            self.buffer_list.append(line)
        if len(self.buffer_list) > self.buffer_size:
            self.flush()

    def write_csv_header(self, print_warning=False):
        if os.path.exists(self.file_to_read_hashes_from):
            if print_warning is True:
                print(f'Попытка перезаписать существующий csv-файл {self.file_to_read_hashes_from}')
            return None
        with open(self.file_to_read_hashes_from, 'w') as fp:
            writer = csv.writer(fp, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(self.header)
        fp.close()

    def __del__(self):
        self.flush()
        print('Остатки буфера записаны в файл')

# Набор всяких околоматематических штук
class Math:

    @staticmethod
    def get_distance(vec1, vec2):
        return scipy.spatial.distance.cosine(vec1, vec2)

    @staticmethod
    def get_hash(hashable):
        hasher = hashlib.sha3_512()
        hasher.update(repr(hashable).encode('utf-8'))
        hash_sum = base64.urlsafe_b64encode(hasher.digest())
        return hash_sum

    @staticmethod
    def sum_vectors(*args):
        res: np.ndarray
        i = 0
        for vec in args:
            if i == 0:
                if len(vec) == 0:
                    vec = [0] * len(args[1])
                res = np.array(vec)
            else:
                res = np.add(np.array(res), np.array(vec))
            i += 1
        # return matutils.unitvec(res)
        return res


class ProgressBar:
    def __init__(self, total=1, epoch_length=None):
        self.__total__ = total
        if epoch_length is None:
            if total > 100000000:
                self.__epoch_length__ = 100000
            elif total >= 10000000:
                self.__epoch_length__ = 50000
            elif total >= 1000000:
                self.__epoch_length__ = 10000
            elif total >= 100000:
                self.__epoch_length__ = 500
            elif total <= 1000:
                self.__epoch_length__ = 10
            else:
                self.__epoch_length__ = total // 100
        else:
            self.__epoch_length__ = int(epoch_length)
        if self.__epoch_length__ == 0:
            self.__epoch_length__ = 1
        self.__iteration__ = 0
        self.__operations_done__ = 0
        self.__start_time__ = datetime.now()
        self.__timer_mode__ = False

    def sec_to_hours(self, seconds):
        a = str(seconds // 3600)
        b = str((seconds % 3600) // 60)
        c = str((seconds % 3600) % 60)
        d = "{} hours {} mins {} seconds".format(a, b, c)
        return d

    def print_progress_bar(self):
        self.__print_progress_bar__()

    def __print_progress_bar__(self, decimals=1, length=100, fill='█'):
        self.__iteration__ = self.__iteration__ + 1
        self.__operations_done__ = self.__operations_done__ + 1
        if self.__iteration__ >= self.__epoch_length__ or self.__operations_done__ == 1 \
                or self.__operations_done__ >= self.__total__:
            iteration = self.__iteration__
            total = self.__total__
            operations_done = self.__operations_done__

            now_time = datetime.now()
            dt = now_time - self.__start_time__
            avg_iter_duration = (dt.total_seconds() / operations_done)
            est_time = round((total - operations_done) * avg_iter_duration)
            est_time = str(self.sec_to_hours(est_time))

            if self.__operations_done__ == 1:
                est_time = 'неизвестно сколько, нужно ещё поработать, чтобы собрать статистику'

            percent = ("{0:." + str(decimals) + "f}").format(100 * (operations_done / float(total)))
            filledLength = int(length * operations_done // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            suffix = f'Выполнено {operations_done} операций из {int(total)}, при такой скорости осталось ещё {est_time}.'
            print(f'\r|{bar}| {percent}% {suffix}')
            # Print New Line on Complete
            if iteration == total:
                print()
                self.__init__()
            elif self.__operations_done__ != 1:
                self.__iteration__ = 0


class StopTimer:

    def __init__(self, duration=None, tick=None, end_time=None):
        if not (duration is None) and not (end_time is None):
            raise ('Нужно выбрать либо длительность, либо время окончания')
        self.__iteration__ = 0
        self.__operations_done__ = 0
        self.__start_time__ = datetime.now()

        if not (end_time is None):
            t = datetime.strptime(end_time, "%H:%M:%S")

            if (t.hour <= self.__start_time__.hour):  # and (t.minute <= self.__start_time__.minute):
                next_day = datetime.now() + timedelta(days=1)
                year = next_day.year
                month = next_day.month
                day = next_day.day
            else:
                year = self.__start_time__.year
                month = self.__start_time__.month
                day = self.__start_time__.day

            t = t.replace(year=year, day=day, month=month)

            d = t - self.__start_time__
            t_hours, t_minutes, t_seconds = self.__timedelta_to_hours_minutes_seconds__(d)
            self.__max_duration__ = timedelta(hours=t_hours, minutes=t_minutes, seconds=t_seconds)
        else:
            t = datetime.strptime(duration, "%H:%M:%S")
            self.__max_duration__ = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

        t = datetime.strptime(tick, "%H:%M:%S")
        self.__tick__ = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        self.__total_ticks_count__ = round(self.__max_duration__ / self.__tick__)

        self.__ticks_gone__ = 0
        self.__ticks_printed__ = 0

    def __timedelta_to_hours_minutes_seconds__(self, td):
        seconds = td.total_seconds()
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = round(seconds % 60, 1)
        return hours, minutes, seconds

    def check_time_has_gone(self):
        fill = '█'
        if datetime.now() - self.__start_time__ > self.__max_duration__ + self.__tick__:
            print(f'Фактическое количество операций:  {self.__operations_done__}')
            return True
        else:
            self.__ticks_gone__ = (datetime.now() - self.__start_time__).total_seconds() / self.__tick__.total_seconds()
            if (self.__operations_done__ == 0) or self.__ticks_gone__ > self.__ticks_printed__:
                self.__ticks_printed__ = self.__ticks_printed__ + 1

                operations_per_tick = self.__operations_done__ / self.__ticks_gone__
                operation_prognosis = round(operations_per_tick * self.__total_ticks_count__)

                time_gone = datetime.now() - self.__start_time__
                hours, minutes, seconds = self.__timedelta_to_hours_minutes_seconds__(time_gone)
                t_hours, t_minutes, t_seconds = self.__timedelta_to_hours_minutes_seconds__(self.__max_duration__)

                suffix = f'Прошло {hours}::{minutes}::{seconds} из {t_hours}::{t_minutes}::{t_seconds}. '
                suffix = suffix + f'Выполнено {self.__operations_done__} операций. Прогноз {operation_prognosis}'

                percent = ("{0:." + str(1) + "f}").format(
                    100 * (self.__ticks_gone__ / float(self.__total_ticks_count__)))
                filledLength = int(100 * self.__ticks_gone__ // self.__total_ticks_count__)
                bar = fill * filledLength + '-' * (100 - filledLength)
                print(f'\r|{bar}| {percent}% {suffix}')

        self.__operations_done__ = self.__operations_done__ + 1

# TODO почитать про model.eval() и, возможно, заморозить модели или сделать выборку, по которой они будут подбирать константы
# TODO реализовать проверку валидности цепочек в моделях и кворуме


# TODO: в граф — словарь, чтобы по многу раз не считать векторы
# TODO: сохранение графа — super.save + словарь отдельно, не через pickle, чтобы можно было переписывать код класса
# TODO: каждой вершине — простые атрибуты: сколько раз бралась эвристиками в рассмотрение, сколько раз объясняется, в скольки объяснениях участвует
