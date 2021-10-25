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
        self.evt_type_embed_dim = conf.get('evt_type_embed_dim', 8)
        self.seg_hidden_dim = conf.get('seg_hidden_dim', 300)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-uncased')
        self.use_gpu = conf.get('use_gpu', False)
        self.event_schema = conf.get("event_schema", {})
        self.role_list = conf.get("role_list", [])
        self.role_num = len(self.role_list)
        self.event_list = conf.get("event_list", [])
        self.conf = conf

        self.bert = BertModel.from_pretrained(self.pretrained_model_name)

        # self.compress_embed_layer = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.compressd_embed_dim),
        #     nn.ReLU()
        # )
        self.evt_type_embed_layer = nn.Embedding(self.event_num, self.evt_type_embed_dim)

        self.arg_event_concat_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + self.evt_type_embed_dim, self.merged_embed_dim),
            nn.ReLU()
        )

        self.arg_classification_layer = nn.Sequential(
            nn.Linear(self.merged_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.role_num)
        )

        self.egp = EventGraphPropagationLayer(self.embed_dim, self.event_num, self.role_num, self.max_ent_len, self.use_gpu, self.event_schema, self.role_list, self.event_list)
        
    def forward(self, inputs, is_predict=False, use_scorer=True):
        sent_token_id_list, attention_mask_id_list, evt_type_list, evt_mention_mask_list, arg_padding_num_list, arg_padding_mask_list, arg_mask_list, arg_type_list = inputs
        sent_embeds = self.bert(input_ids=sent_token_id_list, attention_mask=attention_mask_id_list)[0]

        evt_type_embeddings = self.evt_type_embed_layer(evt_type_list)

        # batch * max_role_len * 768
        arg_mention_embeds = torch.bmm(arg_mask_list, sent_embeds)
        
        # batch * 1 * 768
        event_mention_embed = torch.bmm(evt_mention_mask_list, sent_embeds)

        # batch * max_role_len * merged_embed_dim
        concated_embeds = torch.cat((event_mention_embed, evt_type_embeddings.unsqueeze(1)), dim=-1)
        concated_embeds = torch.cat((arg_mention_embeds, concated_embeds.expand(-1, self.max_ent_len, -1)), dim=-1)
        merged_arg_embeds = self.arg_event_concat_layer(concated_embeds)

        role_logits = self.arg_classification_layer(merged_arg_embeds)

        if not is_predict:
            pred_score, golden_score = self.egp(role_logits, event_mention_embed, arg_mention_embeds, evt_type_list, arg_type_list, arg_padding_mask_list)
            return role_logits, pred_score, golden_score
        
        pred = self.egp.predict(role_logits, event_mention_embed, arg_mention_embeds, evt_type_list, arg_padding_mask_list, use_scorer=use_scorer)
        return pred
        # return role_logits


class EventGraphPropagationLayer(nn.Module):
    def __init__(self, input_dim, event_num, role_num, max_ent_len, use_gpu, event_schema, role_list, event_list):
        super(EventGraphPropagationLayer, self).__init__()
        self.input_dim = input_dim
        self.role_num = role_num
        self.event_num = event_num
        self.max_ent_len = max_ent_len
        self.use_gpu = use_gpu
        self.event_schema = event_schema
        self.role_list = role_list
        self.event_list = event_list

        # self.WRT = nn.parameter.Parameter(torch.FloatTensor(role_num, input_dim, input_dim))
        # nn.init.kaiming_uniform_(self.WRT, a=math.sqrt(self.role_num))

        # self.WTT = nn.parameter.Parameter(torch.FloatTensor(event_num, input_dim, input_dim))
        # nn.init.kaiming_uniform_(self.WTT, a=math.sqrt(self.event_num))
        self.evt_type_embed_layer = nn.Embedding(self.event_num, input_dim)
        self.arg_type_embed_layer = nn.Embedding(len(self.role_list), input_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.scorer = torch.nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def graph_scorer(self, logits, src_evt_embeddings, src_arg_embeddings, evt_type_list, arg_padding_mask_list):
        # 8 * 10 * 768
        # new_evt_embeddings = src_evt_embeddings.clone().detach()
        # new_arg_embeddings = src_arg_embeddings.clone().detach()
        pred_logits = logits.clone().detach()
        preds = torch.argmax(pred_logits, dim=-1)
        arg_role_embedding = self.arg_type_embed_layer(preds).view(-1, self.max_ent_len, 768)
        evt_type_embedding = self.evt_type_embed_layer(evt_type_list).unsqueeze(1)

        evt_embeddings = evt_type_embedding + src_evt_embeddings
        arg_embeddings = arg_role_embedding + src_arg_embeddings

        evt_seq = torch.cat((evt_embeddings, arg_embeddings), dim=1)
        mask = torch.ones(evt_type_list.size()[0], 1).float()
        if self.use_gpu:
            mask = mask.cuda()
        mask = torch.cat((mask, arg_padding_mask_list), dim=1).bool()
        out = self.transformer_encoder(evt_seq.permute(1, 0, 2), src_key_padding_mask=~mask)
        out = out.permute(1, 0, 2)[:, 0, :]
        graph_score = self.scorer(out)
        return graph_score

        # logits = logits.float()
        # role_evt_trans_matrix_list = torch.mm(logits.view(-1, self.role_num), self.WRT.view(self.role_num, -1))

        # evt_logits = torch.zeros_like(evt_type_list.unsqueeze(-1).expand(-1, self.event_num)).scatter(1, evt_type_list.unsqueeze(-1), 1).float()
        # if self.use_gpu:
        #     evt_logits = evt_logits.cuda()

        # evt_to_evt_matrix_list = torch.mm(evt_logits, self.WTT.view(self.event_num, -1)).view(-1, self.input_dim, self.input_dim)
        
        # # propagation
        # # b * 1 * 768, b * input * input --> b * 1 * 768
        # evt_trans_evt_embedding = torch.bmm(evt_embeddings, evt_to_evt_matrix_list)

        # # (b * max_role_len * input_dim) * (b * max_role_len * input_dim * input_dim) -> (b, self.max_role_len, input_dim)
        # role_trans_evt_embedding = torch.bmm(arg_embeddings.reshape(-1, 1, self.input_dim), role_evt_trans_matrix_list.view(-1, self.input_dim, self.input_dim))
        # role_trans_evt_embedding = role_trans_evt_embedding.view(-1, self.max_ent_len, self.input_dim)

        # # update
        # mask = arg_padding_mask_list.unsqueeze(-1).expand(-1, -1, self.input_dim)
        # role_trans_evt_embedding *= mask
        # # # e1 = wt * e0 + avg(sum(wri * ai)) -> 1 * out

        # new_role_embeds_sum = torch.sum(role_trans_evt_embedding, dim=1)
        # arg_padding_sum = torch.sum(arg_padding_mask_list, dim=-1).unsqueeze(-1)
        # div = torch.div(new_role_embeds_sum, arg_padding_sum)
        # graph_embeddings = 0.8 * div + 0.2 * evt_trans_evt_embedding.squeeze()
        # graph_score = self.scorer(graph_embeddings)
        # return graph_score

    def forward(self, role_logits, evt_embeddings, arg_embeddings, evt_type_list, arg_type_list, arg_padding_mask):
        evt_type_list_np = evt_type_list.cpu().numpy()

        # batch * max_role_len * role_num
        logits = role_logits.clone().detach()
        logits = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        candidate_preds = []
        candidate_ground_truth = []
        indexs = []

        # negtive sampler
        for idx, (ps, ls, et, arg_padding) in enumerate(zip(preds, arg_type_list, evt_type_list_np, arg_padding_mask)):
            ps = ps.cpu()
            ls = ls.cpu()
            
            is_valid = True
            arg_len = 0
            candidate_ground_truth.append(ls.cpu().numpy())
            for p, l, ap in zip(ps, ls, arg_padding):
                if int(ap) == 0:
                    break
                if p != l:
                    is_valid = False
                    candidate_preds.append(ps.clone().detach().cpu().numpy())
                    # candidate_ground_truth.append(ls.cpu().numpy())
                    indexs.append(idx)
                    break
                arg_len += 1
            if is_valid:
                candidates = [self.role_list.index(x) for x in self.event_schema[self.event_list[et]]]
                pivots = set()
                while len(pivots) < arg_len and len(pivots) < 3:
                    cur_ps = ps.clone().detach().cpu().numpy()
                    cur_ls = ls.clone().detach().cpu().numpy()

                    pivot = int(random.random() % 100 * arg_len) % arg_len
                    while pivot in pivots:
                        pivot = int(random.random() % 100 * arg_len) % arg_len
                    pivots.add(pivot)

                    pred_label = ps[pivot]
                    pred_value = ls[pivot]
                    candidate_value = 0
                    candidate_label = 0
                    for i, val in enumerate(logits[idx][pivot]):
                        if float(val) > candidate_value and int(pred_label) != i and i in candidates:
                            candidate_value = float(val)
                            candidate_label = i
                    cur_ps[pivot] = candidate_label
                    candidate_preds.append(cur_ps)
                    # if random.random() < 1:
                    #     print(ps, ls, cur_ps)
                    # candidate_ground_truth.append(cur_ls)
                    indexs.append(idx)
        new_evt_embeddings = torch.zeros((len(indexs), 1, int(evt_embeddings.size()[-1]))).float()
        new_arg_embeddings = torch.zeros((len(indexs), self.max_ent_len, int(arg_embeddings.size()[-1]))).float()
        new_evt_type_list = torch.zeros(len(indexs)).long()
        new_arg_padding_mask = torch.zeros(len(indexs), self.max_ent_len).float()
        if self.use_gpu:
            new_evt_embeddings = new_evt_embeddings.cuda()
            new_arg_embeddings = new_arg_embeddings.cuda()
            new_evt_type_list = new_evt_type_list.cuda()
            new_arg_padding_mask = new_arg_padding_mask.cuda()
        
        for i, index in enumerate(indexs):
            new_evt_embeddings[i] = evt_embeddings[index]
            new_arg_embeddings[i] = arg_embeddings[index]
            new_evt_type_list[i] = evt_type_list[index]
            new_arg_padding_mask[i] = arg_padding_mask[index]

        candidate_preds = torch.LongTensor(candidate_preds)
        candidate_ground_truth = torch.LongTensor(candidate_ground_truth)
        if self.use_gpu:
            candidate_ground_truth = candidate_ground_truth.cuda()
            candidate_preds = candidate_preds.cuda()

        preds = candidate_preds.view(-1, 1)
        pred_logits = torch.zeros_like(preds.expand(-1, self.role_num)).scatter(1, preds, 1)
        pred_logits = pred_logits.view(-1, self.max_ent_len, self.role_num)
        if self.use_gpu:
            pred_logits = pred_logits.cuda()
        pred_graph_score = self.graph_scorer(pred_logits, new_evt_embeddings, new_arg_embeddings, new_evt_type_list, new_arg_padding_mask)
        # pred_graph_score = self.graph_scorer(candidate_preds, new_evt_embeddings, new_arg_embeddings, new_evt_type_list, new_arg_padding_num)

        candidate_ground_truth = candidate_ground_truth.view(-1, 1)
        golden_logits = torch.zeros_like(candidate_ground_truth.expand(-1, self.role_num)).scatter(1, candidate_ground_truth, 1)
        gloden_logits = golden_logits.view(-1, self.max_ent_len, self.role_num)
        if self.use_gpu:
            golden_logits = golden_logits.cuda()
        golden_graph_score = self.graph_scorer(golden_logits, evt_embeddings, arg_embeddings, evt_type_list, arg_padding_mask)
        # golden_graph_score = self.graph_scorer(candidate_ground_truth, evt_embeddings, arg_embeddings, evt_type_list, arg_padding_num)

        return pred_graph_score, golden_graph_score

    def predict(self, role_logits, evt_embeddings, arg_embeddings, evt_type_list, arg_padding_num, use_scorer=True):
        # batch * max_role_len * role_num
        logits = role_logits.detach()
        logits = torch.log_softmax(logits, dim=-1)
        labels = torch.argmax(logits, dim=-1)
        if not use_scorer:
            return labels
        result = []
        for i, (batch_logit, batch_label, batch_evt_types, batch_padding_num) in enumerate(zip(logits, labels, evt_type_list, arg_padding_num)):
            cand_labels = [batch_label.clone().cpu().numpy()]
            evt_types = [batch_evt_types.clone().cpu() for i in range(self.max_ent_len + 1)]
            padding_nums = [batch_padding_num.clone().cpu().numpy() for i in range(self.max_ent_len + 1)]
            for j, (logit, pred) in enumerate(zip(batch_logit, batch_label)):
                logit = torch.softmax(logit, dim=-1)
                max_idx = torch.argmax(logit, dim=-1).cpu().numpy()
                max_val = torch.max(logit, dim=-1).values.cpu().numpy()
                second_idx = 0
                second_val = 0
                for k, val in enumerate(logit):
                    if k == max_idx:
                        continue
                    if float(val) > second_val and float(val) < max_val:
                        second_val = float(val)
                        second_idx = k

                tmp_labels = batch_label.clone()
                tmp_labels[j] = second_idx
                cand_labels.append(tmp_labels.cpu().numpy())
            cand_labels = torch.LongTensor(cand_labels)
            evt_types = torch.LongTensor(evt_types)
            padding_nums = torch.FloatTensor(padding_nums)
            if self.use_gpu:
                cand_labels = cand_labels.cuda()
                evt_types = evt_types.cuda()
                padding_nums = padding_nums.cuda()
            
            preds = cand_labels.view(-1, 1)
            pred_logits = torch.zeros_like(preds.view(-1, 1).expand(-1, self.role_num)).scatter(1, preds, 1)
            pred_logits = pred_logits.view(-1, self.max_ent_len, self.role_num)
            if self.use_gpu:
                pred_logits = pred_logits.cuda()
            tmp_evt_embeddings = evt_embeddings[i].clone().unsqueeze(0).expand(self.max_ent_len+1, -1, -1)
            tmp_arg_embeddings = arg_embeddings[i].clone().unsqueeze(0).expand(self.max_ent_len+1, -1, -1)
            pred_graph_score = self.graph_scorer(pred_logits, tmp_evt_embeddings, tmp_arg_embeddings, evt_types, padding_nums)
            pred_graph_score = torch.softmax(pred_graph_score, dim=-1)
            pred_graph_score = pred_graph_score[:, 1].view(-1)
            res = torch.argmax(pred_graph_score)
            # if random.random() < 0.01:
            #     print(res, pred_graph_score, cand_labels)
            result.append(cand_labels[res].cpu().numpy())

        result = torch.LongTensor(result)
        if self.use_gpu:
            result = result.cuda()

        return result