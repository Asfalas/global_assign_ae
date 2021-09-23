import sys
sys.path.append('./')
import json
import logging

from channels.common.dataset import *
from channels.common.ace05_dataset import *

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(filename='log/test_dataset.log',level=logging.INFO)

conf = json.load(open('conf/ace05/config.json'))

ace_dataset = CommonDataSet('ACE05', ACE05DataHandler, 'data/ace05/ace05_test.json', conf)