import sys
sys.path.append('./')
import math
import torch.nn as nn
import logging
import torch

from transformers import BertModel
from channels.common.const import *
from channels.common.seg_module import SelfEnhancingGraphModule

class EncoderLayer(nn.Module):
    def __init__(self, conf):
        super(EncoderLayer, self).__init__()
        self.embed_dim = conf.get('embed_dim', 768)
        self.lstm_hidden_dim = conf.get('lstm_hidden_dim', 1024)
        self.compressd_embed_dim = conf.get('compressd_embed_dim', 500)
        self.merged_embed_dim = conf.get('merged_embed_dim', 500)
        self.max_role_len = conf.get('max_role_len', 16)
        self.max_ent_len = conf.get('max_ent_len', 28)
        self.event_num = conf.get('evt_num', 34)
        self.seg_hidden_dim = conf.get('seg_hidden_dim', 300)
        self.use_seg = conf.get('use_seg', False)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-uncased')
        self.use_gpu = conf.get('use_gpu', False)
        self.conf = conf

        self.bert = BertModel.from_pretrained(self.pretrained_model_name)
        # Bi_LSTM layer
        # self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_dim // 2, num_layers=1,
        #                              bidirectional=True,
        #                              batch_first=True)
        self.compress_embed_layer = nn.Sequential(
            # nn.Linear(self.lstm_hidden_dim, self.compressd_embed_dim),
            nn.Linear(self.embed_dim, self.compressd_embed_dim),
            nn.ReLU()
        )

    def forward(self, sent_token_id_list, arg_mask_list, evt_mention_mask_list):
        # self.lstm.flatten_parameters()
        # batch * 256 * 768
        sent_embeds = self.bert(input_ids=sent_token_id_list)[0]
        # batch * 256 * 500
        # sent_embeds, (h, c) = self.lstm(sent_embeds)
        # batch * 28 * 256 , batch * 256 * 500 -> batch * 28 * 500
        arg_mention_embeds = torch.bmm(arg_mask_list, sent_embeds)
        arg_mention_embeds = self.compress_embed_layer(arg_mention_embeds)
        # batch * 1 * 512, batch * 512 * 500 -> batch * 1 * 500
        event_mention_embed = torch.bmm(evt_mention_mask_list, sent_embeds)
        event_mention_embed = self.compress_embed_layer(event_mention_embed)

        return arg_mention_embeds, event_mention_embed


class SelfEnhancingGraphEAE(nn.Module):
    def __init__(self, conf):
        super(SelfEnhancingGraphEAE, self).__init__()
        self.embed_dim = conf.get('embed_dim', 768)
        self.lstm_hidden_dim = conf.get('lstm_hidden_dim', 1024)
        self.compressd_embed_dim = conf.get('compressd_embed_dim', 500)
        self.merged_embed_dim = conf.get('merged_embed_dim', 500)
        self.max_role_len = conf.get('max_role_len', 16)
        self.max_ent_len = conf.get('max_ent_len', 28)
        self.event_num = conf.get('evt_num', 34)
        self.seg_hidden_dim = conf.get('seg_hidden_dim', 300)
        self.use_seg = conf.get('use_seg', False)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-uncased')
        self.use_gpu = conf.get('use_gpu', False)
        self.conf = conf

        self.encoder = EncoderLayer(conf)

        self.init_classifier = ArgClassifierLayer(conf)

        self.seg_classifier = ArgClassifierLayer(conf)

        self.seg_module = SelfEnhancingGraphModule(conf)

    def forward(self, inputs):
        sent_token_id_list, evt_type_list, evt_mention_mask_list, arg_padding_mask_list, arg_padding_num_list, arg_mask_list = tuple(
            x for x in inputs)
        evt_type_list = evt_type_list.cpu().numpy()
        arg_padding_num_list = arg_padding_num_list.cpu().numpy()

        arg_mention_embeds, event_mention_embed = self.encoder(sent_token_id_list, arg_mask_list, evt_mention_mask_list)
        # batch * 1 * 200
        # classification
        logits = self.init_classifier(arg_mention_embeds, event_mention_embed, evt_type_list)
        if self.use_seg:
            count= 0
            max_iter_num = self.conf.get('max_iter_num', 0)
            while count < max_iter_num:
                # arg_mention_embeds, event_mention_embed = self.seg_module(logits, arg_mention_embeds, event_mention_embed, arg_padding_num_list, evt_type_list)
                new_logits = self.seg_classifier(arg_mention_embeds, event_mention_embed, evt_type_list)

                old_res = torch.argmax(torch.softmax(logits.clone().detach(), dim=-1), dim=-1)
                new_res = torch.argmax(torch.softmax(new_logits.clone().detach(), dim=-1), dim=-1)
                logits = new_logits
                if old_res.equal(new_res):
                    break
                count += 1
        return logits


class ArgClassifierLayer(nn.Module):
    def __init__(self, conf):
        super(ArgClassifierLayer, self).__init__()
        self.embed_dim = conf.get('lstm_hidden_dim', 600)
        self.merged_embed_dim = conf.get('merged_embed_dim', 500)
        self.compressd_embed_dim = conf.get('compressd_embed_dim', 500)
        self.max_role_len = conf.get('max_role_len', 16)
        self.max_ent_len = conf.get('max_ent_len', 28)
        self.event_num = conf.get('evt_num', 34)
        self.seg_hidden_dim = conf.get('seg_hidden_dim', 300)
        self.use_seg = conf.get('use_seg', False)
        self.max_iter_num = conf.get('max_iter_num', 1)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-uncased')
        self.use_gpu = conf.get('use_gpu', False)

        self.mention_merge_layer = nn.Sequential(
            nn.Linear(self.compressd_embed_dim * 2, self.merged_embed_dim),
            nn.ReLU()
        )

        self.arg_classification_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.merged_embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.max_role_len)
            )
            for i in range(self.event_num)]
        )

    def forward(self, arg_mention_embeds, event_mention_embed, evt_type_list):
        # merge & confuse
        merged_mention_embeds = torch.cat((arg_mention_embeds, event_mention_embed.expand(-1, self.max_ent_len, -1)),
                                       dim=-1)
        merged_mention_embeds = self.mention_merge_layer(merged_mention_embeds)

        # classifier
        logits = torch.FloatTensor(len(evt_type_list), self.max_ent_len, self.max_role_len)
        if self.use_gpu:
            logits = logits.cuda()
        for idx, event_id in enumerate(evt_type_list):
            event_id = evt_type_list[idx]
            logits[idx: idx + 1] = self.arg_classification_layers[event_id](merged_mention_embeds[idx: idx + 1])
        return logits