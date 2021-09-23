import sys
sys.path.append('./')
import json
import copy
import torch
import logging

from channels.common.dataset import *
from tqdm import tqdm

class ACE05DataHandler(CommonDataHandler):
    def _load_data(self):
        # for output tensor
        sent_token_id_list = []
        attention_mask_id_list = []
        evt_mention_mask_list = []
        arg_padding_mask_list = []
        evt_type_list = []
        arg_mask_list = []
        arg_type_list = []
        arg_padding_num_list = []

        arg_err = 0
        evt_err = 0
        empty_arg_err = 0
        
        for event_info in tqdm(self.data):
            tokens = event_info['tokens']
            token_ids, flatten_token_ids, token_offsets, attention_mask_ids = self.sentence_padding(tokens)
            event_type = event_info['evt_type']

            # calc event offsets
            tmp_evt_offset = self.get_flatten_offsets(token_offsets, [event_info['evt_beg'], event_info['evt_end']])
            valid = True
            for offset in tmp_evt_offset:
                if offset >= self.max_seq_len:
                    valid = False
                    evt_err += 1
                    break
            if not valid:
                continue
            assert flatten_token_ids[tmp_evt_offset[0] : tmp_evt_offset[1]+1] == reduce(lambda x, y: x+y, token_ids[event_info['evt_beg']: event_info['evt_end']+1])

            # event mention mask
            tmp_evt_mask = [0.0] * self.max_seq_len
            for i in range(tmp_evt_offset[0], tmp_evt_offset[1] + 1):
                tmp_evt_mask[i] = 1.0 / (tmp_evt_offset[1] + 1 - tmp_evt_offset[0])

            # arg mention mask & arg type
            arg_mention_mask_template = [0.0] * self.max_seq_len
            tmp_arg_masks = []
            tmp_arg_types = []
            tmp_arg_padding_mask = []

            valid = True
            if len(event_info.get('args', [])) == 0:
                empty_arg_err += 1
                continue

            for arg in event_info['args']:
                offsets = [arg['arg_beg'], arg['arg_end']]
                new_offsets = self.get_flatten_offsets(token_offsets, offsets)

                valid = True
                for offset in new_offsets:
                    if offset >= self.max_seq_len:
                        valid = False
                        arg_err += 1
                        break
                if not valid:
                    continue

                tmp_offsets = copy.copy(arg_mention_mask_template)
                for i in range(new_offsets[0], new_offsets[1] + 1):
                    tmp_offsets[i] = 1.0 / (new_offsets[1] + 1 - new_offsets[0])
                tmp_arg_masks.append(tmp_offsets)

                arg_type = arg['role']
                # tmp_arg_types.append(self.event_schema[event_type].index(arg_type))
                tmp_arg_types.append(self.role_list.index(arg_type))
                tmp_arg_padding_mask.append([1] * self.max_role_len)

            for i in range(len(tmp_arg_types), self.max_ent_len):
                tmp_arg_types += [-100]
                tmp_arg_masks += copy.copy([arg_mention_mask_template])
                tmp_arg_padding_mask += [[0] * self.max_role_len]

            # event type mask
            tmp_event_type_mask = [0] * self.evt_num
            tmp_event_type_mask[self.event_list.index(event_type)] = 1

            # put in output tensor
            sent_token_id_list.append(flatten_token_ids)
            attention_mask_id_list.append(attention_mask_ids)
            print(len(flatten_token_ids), len(attention_mask_ids))
            arg_type_list.append(tmp_arg_types)
            arg_mask_list.append(tmp_arg_masks)
            arg_padding_mask_list.append(tmp_arg_padding_mask)
            padding_list = [1.0 if i < len(event_info['args']) else 0.0 for i in range(self.max_ent_len)]
            arg_padding_num_list.append(len(event_info['args']))
            evt_mention_mask_list.append([copy.copy(tmp_evt_mask)])
            evt_type_list.append(self.event_list.index(event_type))

        # transform to tensor
        sent_token_id_list = torch.LongTensor(sent_token_id_list)
        attention_mask_id_list = torch.LongTensor(attention_mask_id_list)
        arg_type_list = torch.LongTensor(arg_type_list)
        arg_mask_list = torch.FloatTensor(arg_mask_list)
        arg_padding_mask_list = torch.FloatTensor(arg_padding_mask_list)
        arg_padding_num_list = torch.FloatTensor(arg_padding_num_list)
        evt_type_list = torch.LongTensor(evt_type_list)
        evt_mention_mask_list = torch.FloatTensor(evt_mention_mask_list)

        if evt_err:
            logging.warn("  触发词越界: " + str(evt_err))
        if empty_arg_err:
            logging.info("  空论元列表: " + str(empty_arg_err))
        if arg_err:
            logging.info("  论元越界: " + str(arg_err))
        
        return sent_token_id_list, attention_mask_id_list, evt_type_list, evt_mention_mask_list, arg_padding_mask_list, arg_padding_num_list, arg_mask_list, arg_type_list
