import sys
sys.path.append('./')
import time
import torch
import json
import logging

from tqdm import tqdm
from channels.common.const import *
from channels.common.utils import *
from channels.common.processor import *
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.optim import lr_scheduler

class GlobalAssignerProcessor(Processor):
    def __init__(self):
        super(GlobalAssignerProcessor, self).__init__()

    def setup(self, args):
        total_params = self.model.parameters() if not self.use_gpu else self.model.module.parameters()
        self.optimizer = torch.optim.Adam(total_params, lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_gamma)
        self.loss_function = nn.CrossEntropyLoss()