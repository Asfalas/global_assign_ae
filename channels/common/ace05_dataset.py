import sys
sys.path.append('./')
import json
import copy
import torch
import logging
import random

from channels.common.dataset import *
from tqdm import tqdm

class ACE05DataHandler(CommonDataHandler):
    def _load_data(self):
        # for output tensor
        sent_token_id_list = []
        attention_mask_id_list = []
        evt_mention_mask_list = []
        evt_type_list = []
        arg_mask_list = []
        arg_type_list = []
        arg_padding_num_list = []
        role_label_list = []

        arg_err = 0
        evt_err = 0
        empty_arg_err = 0
        arg_num_err = 0
        n = 0
        t = 0
        
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
            tmp_arg_labels = []

            valid = True
            if len(event_info.get('args', [])) == 0:
                empty_arg_err += 1
                continue

            none_args = []
            not_padding_arg_num = 0
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
                
                assert flatten_token_ids[new_offsets[0] : new_offsets[1]+1] == reduce(lambda x, y: x+y, token_ids[arg['arg_beg']: arg['arg_end']+1])

                tmp_offsets = copy.copy(arg_mention_mask_template)
                for i in range(new_offsets[0], new_offsets[1] + 1):
                    tmp_offsets[i] = 1.0 / (new_offsets[1] + 1 - new_offsets[0])

                arg_type = arg['role']
                if arg['role'] == 'None':
                    none_args.append(tmp_offsets)
                    continue
                t += 1
                not_padding_arg_num += 1
                # tmp_arg_types.append(self.event_schema[event_type].index(arg_type))
                tmp_arg_types.append(self.role_list.index(arg_type))
                tmp_arg_labels.append(self.role_list.index(arg_type))
                tmp_arg_masks.append(tmp_offsets)
            
            if len(tmp_arg_types) > self.max_ent_len:
                arg_num_err += 1
                tmp_arg_types = tmp_arg_types[:self.max_ent_len]
                tmp_arg_labels = tmp_arg_labels[:self.max_ent_len]
                tmp_arg_masks = tmp_arg_masks[:self.max_ent_len]

            random.shuffle(none_args)
            none_args = none_args[:self.max_ent_len - len(tmp_arg_types)]
            for n_arg in none_args:
                n += 1
                tmp_arg_types.append(0)
                tmp_arg_labels.append(0)
                tmp_arg_masks.append(n_arg)
                not_padding_arg_num += 1

            for i in range(len(tmp_arg_types), self.max_ent_len):
                tmp_arg_types += [0]
                tmp_arg_labels += [-100]
                tmp_arg_masks += copy.copy([arg_mention_mask_template])
            
            # event type mask
            tmp_event_type_mask = [0] * self.evt_num
            tmp_event_type_mask[self.event_list.index(event_type)] = 1

            # put in output tensor
            sent_token_id_list.append(flatten_token_ids)
            attention_mask_id_list.append(attention_mask_ids)
            arg_type_list.append(tmp_arg_types)
            role_label_list.append(tmp_arg_labels)
            arg_mask_list.append(tmp_arg_masks)
            arg_padding_num_list.append(not_padding_arg_num)
            evt_mention_mask_list.append([copy.copy(tmp_evt_mask)])
            evt_type_list.append(self.event_list.index(event_type))

        # transform to tensor
        sent_token_id_list = torch.LongTensor(sent_token_id_list)
        attention_mask_id_list = torch.LongTensor(attention_mask_id_list)
        arg_type_list = torch.LongTensor(arg_type_list)
        role_label_list = torch.LongTensor(role_label_list)
        arg_mask_list = torch.FloatTensor(arg_mask_list)
        arg_padding_num_list = torch.FloatTensor(arg_padding_num_list)
        evt_type_list = torch.LongTensor(evt_type_list)
        evt_mention_mask_list = torch.FloatTensor(evt_mention_mask_list)

        if evt_err:
            logging.warn("  触发词越界: " + str(evt_err))
        if empty_arg_err:
            logging.info("  空论元列表: " + str(empty_arg_err))
        if arg_err:
            logging.info("  论元越界: " + str(arg_err))
        if arg_num_err:
            logging.info("  论元数目越界: " + str(arg_num_err))
        print(n, t)
        
        return sent_token_id_list, attention_mask_id_list, evt_type_list, evt_mention_mask_list, arg_padding_num_list, arg_mask_list, arg_type_list, role_label_list
