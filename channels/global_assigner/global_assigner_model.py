import sys
sys.path.append('./')
import math
import torch.nn as nn
import logging
import torch
import random

from transformers import BertModel
from channels.common.const import *

class GlobalAssigner(nn.Module):
    def __init__(self, conf):
        super(GlobalAssigner, self).__init__()
        self.embed_dim = conf.get('embed_dim', 768)
        self.compressd_embed_dim = conf.get('compressd_embed_dim', 500)
        self.merged_embed_dim = conf.get('merged_embed_dim', 500)
        self.max_role_len = conf.get('max_role_len', 16)
        self.max_ent_len = conf.get('max_ent_len', 28)
        self.event_num = conf.get('evt_num', 34)
        self.seg_hidden_dim = conf.get('seg_hidden_dim', 300)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-uncased')
        self.use_gpu = conf.get('use_gpu', False)
        self.event_schema = conf.get("event_schema", {})
        self.role_list = conf.get("role_list", [])
        self.role_num = len(self.role_list)
        self.event_list = conf.get("event_list", [])
        self.conf = conf

        self.bert = BertModel.from_pretrained(self.pretrained_model_name)

        self.compress_embed_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.compressd_embed_dim),
            nn.ReLU()
        )

        self.arg_event_concat_layer = nn.Sequential(
            nn.Linear(self.compressd_embed_dim * 2, self.merged_embed_dim),
            nn.ReLU()
        )

        self.arg_classification_layer = nn.Sequential(
            nn.Linear(self.merged_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.role_num)
        )

        self.egp = EventGraphPropagationLayer(compressd_embed_dim, event_num, role_num, max_role_len, use_gpu, event_schema, role_list)
        
    def forward(self, inputs):
        sent_token_id_list, attention_mask_id_list, evt_type_list, evt_mention_mask_list, arg_padding_mask_list, arg_padding_num_list, arg_mask_list, arg_type_list = inputs
        sent_embeds = self.bert(input_ids=sent_token_id_list)[0]

        # batch * max_role_len * 768
        arg_mention_embeds = torch.bmm(arg_mask_list, sent_embeds)
        arg_mention_embeds = self.compress_embed_layer(arg_mention_embeds)
        
        # batch * 1 * 768
        event_mention_embed = torch.bmm(evt_mention_mask_list, sent_embeds)
        event_mention_embed = self.compress_embed_layer(event_mention_embed)

        # batch * max_role_len * merged_embed_dim
        concated_embeds = torch.cat((arg_mention_embeds, event_mention_embed.expand(-1, self.max_role_len, -1)), dim=-1)
        merged_arg_embeds = self.arg_event_concat_layer(concated_embed)

        role_logits = self.arg_classification_layer(merged_arg_embeds)

        pred_score, golden_score = self.egp(role_logits, event_mention_embed, arg_mention_embeds, evt_type_list, arg_type_list, arg_padding_num_list)

        return role_logits


class EventGraphPropagationLayer(nn.Module):
    def __init__(self, input_dim, event_num, role_num, max_role_len, use_gpu, event_schema, role_list):
        super(EventGraphPropagationLayer, self).__init__()
        self.input_dim = input_dim
        self.role_num = role_num
        self.event_num = event_num
        self.max_role_len = max_role_len
        self.use_gpu = use_gpu
        self.event_schema = event_schema
        self.role_list = role_list

        self.WRT = nn.parameter.Parameter(torch.FloatTensor(role_num, input_dim, input_dim))
        nn.init.kaiming_uniform_(self.WRT, a=math.sqrt(5))

        self.WTT = nn.parameter.Parameter(torch.FloatTensor(event_num, input_dim, input_dim))
        nn.init.kaiming_uniform_(self.WTT, a=math.sqrt(5))

        self.scorer = self.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def graph_scorer(self, logits, evt_embeddings, arg_embeddings, evt_type_list, arg_type_list, arg_padding_num):
        role_evt_trans_matrix_list = torch.mm(logits.view(-1, self.role_num), self.WRT.view(self.role_num, -1))

        evt_logits = torch.zeros_like(evt_type_list.unsqueeze(-1).expand(-1, self.event_num)).scatter_(1, evt_type_list.unsqueeze(-1), 1)
        evt_to_evt_matrix_list = torch.mm(evt_logits, self.WTT.view(self.event_num, -1)).view(-1, self.input_dim, self.input_dim)
        
        # propagation
        # b * 1 * 768, b * input * input --> b * 1 * 768
        evt_trans_evt_embedding = torch.bmm(evt_embeddings, evt_to_evt_matrix_list)

        # (b * max_role_len * input_dim) * (b * max_role_len * input_dim * input_dim) -> (b, self.max_role_len, input_dim)
        role_trans_evt_embedding = torch.bmm(arg_embeddings.view(-1, 1, input_dim), role_evt_trans_matrix_list.view(-1, input_dim, input_dim))
        role_trans_evt_embedding = role_trans_evt_embedding.view(-1, self.max_role_len, input_dim)

        # update
        role_trans_evt_embedding *= arg_padding_num.unsqueeze(-1).expand(-1, -1, input_dim)
        # # e1 = wt * e0 + avg(sum(wri * ai)) -> 1 * out
        graph_embeddings = (torch.div(torch.sum(role_trans_evt_embedding, dim=1),
                                        torch.sum(arg_padding_num, dim=1).unsqueeze(-1)) + evt_trans_evt_embedding.squeeze()) / 2.0
        graph_score = self.scorer(graph_embeddings)
        return graph_score

    def forward(self, role_logits, evt_embeddings, arg_embeddings, evt_type_list, arg_type_list, arg_padding_num):
        evt_type_list = evt_type_list.cpu().numpy()
        arg_type_list = arg_type_list

        # batch * max_role_len * role_num
        logits = role_logits.detach()
        logits = torch.log_softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        # negtive sampler
        for idx, (ps, ls, et) in enumerate(zip(preds, arg_type_list, event_type_list)):
            is_valid = True
            arg_len = 0
            for p, l in zip(ps, ls):
                if l == -100:
                    break
                if p != l:
                    is_valid = False
                    break
                arg_len += 1
            if is_valid:
                candidates = self.event_schema[self.event_list[et]]
                pivot = 0
                cand = 0
                while ps[pivot] == self.role_list.index(candidates[cand]):
                    pivot = int(random.random()) % arg_len
                    cand = int(random.random()) % len(candidates)
                preds[idx][pivot] = self.role_list.index(candidates[cand])

        pred_logits = torch.zeros_like(role_logits.view(-1, self.role_num)).scatter_(1, preds.view(-1), 1)
        pred_logits = pred_logits.view(-1, self.max_role_len, self.role_num)
        if self.use_gpu:
            pred_logits = pred_logits.cuda()
        pred_graph_score = self.graph_scorer(pred_logits, evt_embeddings, arg_embeddings, evt_type_list, arg_type_list, arg_padding_num)

        golden_logits = torch.zeros_like(arg_type_list.view(-1).unsqueeze(-1).expand(-1, self.role_num)).scatter_(1, arg_type_list.view(-1), 1)
        golden_graph_score = self.graph_scorer(golden_logits, evt_embeddings, arg_embeddings, evt_type_list, arg_type_list, arg_padding_num)

        return pred_graph_score, golden_graph_score