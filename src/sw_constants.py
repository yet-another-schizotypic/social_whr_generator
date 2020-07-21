import logging
import os
import pathlib
LOGLEVEL = logging.INFO

SW_CONFIG_FILE_NAME = 'sw_config.json'
SW_SCRIPT_PATH = pathlib.Path(__file__).parent.absolute()
SW_SCRIPT_OUTPUT_PATH = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), 'output/')