import sys
sys.path.append('./')
import json
import logging

from transformers import BertTokenizer
from torch.utils.data import Dataset
from _functools import reduce

class CommonDataSet(Dataset):
    def __init__(self, dataset_name, data_handler, path, conf, debug=0):
        logging.info('  数据处理开始: ' + dataset_name + ' --> ' + path)
        dh = data_handler(path=path, conf=conf, debug=debug)
        self.data = dh.load()
        logging.info('  数据处理结束!\n')

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return tuple(x[idx] for x in self.data)

class CommonDataHandler(object):
    def __init__(self, path, conf, debug=0):
        self.path = path
        self.data = json.load(open(path))
        logging.info('  debug 模式:' + ("True" if debug else "False"))
        if debug:
            self.data = self.data[:200]
        self.event_schema = conf.get('event_schema', {})
        self.event_list = conf.get('event_list', [])
        self.role_list = conf.get('role_list', [])
        self.max_seq_len = conf.get('max_seq_len', 256)
        self.max_ent_len = conf.get('max_ent_len', 28)
        self.max_role_len = conf.get('max_role_len', 16)
        self.evt_num = conf.get('evt_num', 34)
        self.add_cls = conf.get('add_cls', True)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)

    def get_flatten_offsets(self, token_ids, old_offsets):
        old_start, old_end = old_offsets[0], old_offsets[1]
        new_start = token_ids[old_start][0]
        new_end = token_ids[old_end][-1]
        return [new_start + 1, new_end + 1] if self.add_cls else [new_start, new_end]

    def convert_tokens_to_ids(self, tokens):
        input_ids = []
        ids = self.tokenizer(tokens)['input_ids']
        for id in ids:
            input_ids.append(id[1:-1])

        return input_ids

    def sentence_padding(self, tokens):
        token_ids = self.convert_tokens_to_ids(tokens)
        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")

        token_offsets = []

        offset = 0
        for i in token_ids:
            token_offsets.append([offset, offset + len(i) - 1])
            offset += len(i)

        flatten_token_ids = reduce(lambda x, y: x + y, token_ids)
        attention_mask_ids = [1 for i in flatten_token_ids]

        if len(flatten_token_ids) <= self.max_seq_len:
            flatten_token_ids = flatten_token_ids + [pad_id] * (self.max_seq_len - len(flatten_token_ids))
            attention_mask_ids = attention_mask_ids + [0] * (self.max_seq_len - len(flatten_token_ids))
        else:
            flatten_token_ids = flatten_token_ids[:self.max_seq_len]
            attention_mask_ids = attention_mask_ids[:self.max_seq_len]
        if self.add_cls:
            flatten_token_ids = [cls_id] + flatten_token_ids[:self.max_seq_len - 1]
            attention_mask_ids = [1] + attention_mask_ids[:self.max_seq_len - 1]
        
        return token_ids, flatten_token_ids, token_offsets, attention_mask_ids

    def _load_data(self):
        raise Exception('not implement error!')

    def load(self):
        return self._load_data()
