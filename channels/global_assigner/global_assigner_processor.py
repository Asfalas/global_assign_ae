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

class GlobalAssignerProcessor(CommonModelProcessor):
    def __init__(self, model, train_dataset, dev_dataset, test_dataset, conf):
        super(GlobalAssignerProcessor, self).__init__(model, train_dataset, dev_dataset, test_dataset, conf)
        self.transformer_lr = conf.get("transformer_lr", 5e-4)

    def setup(self):
        if not self.conf.get("train_bert", 0):
            logging.info("  锁定bert参数！")
            bert_params = self.model.bert.parameters() if not self.use_gpu else self.model.module.bert.parameters()
            for i in bert_params:
                i.requires_grad = False
        total_params = self.model.parameters() if not self.use_gpu else self.model.module.parameters()
        self.optimizer = torch.optim.Adam(total_params, lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_gamma)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100, size_average=True)
        self.margin_loss_function = nn.MarginRankingLoss(margin=0.3)
        self.score_loss_function = nn.CrossEntropyLoss(ignore_index=-100, size_average=True)

    def get_score_loss(self, pred_scores, golden_scores):
        pred_scores = pred_scores.view(-1, 2)
        golden_scores = golden_scores.view(-1, 2)

        scores = torch.cat((pred_scores, golden_scores), dim=0)
        y1 = torch.zeros(pred_scores.size()[0])
        y2 = torch.ones(golden_scores.size()[0])
        y = torch.cat((y1, y2), dim=0).long()
        if self.use_gpu:
            y = y.cuda()
        loss = self.score_loss_function(scores, y)
        return loss

    def get_margin_loss(self, pred_scores, golden_scores):
        pred_scores = pred_scores.view(-1)
        golden_score = golden_scores.view(-1)
        y = torch.ones_like(pred_scores)
        if self.use_gpu:
            y = y.cuda()
        loss = self.margin_loss_function(golden_scores, pred_scores, y)
        return loss

    def get_classify_loss(self, pred, label):
        label = label.view(-1, 1).squeeze()
        pred = pred.view(label.size()[0], -1)
        loss = self.loss_function(pred, label)
        return loss

    def get_loss(self, pred, pred_scores, golden_scores, label):
        # loss1 = self.get_margin_loss(pred_scores, golden_scores)
        loss2 = self.get_classify_loss(pred, label)
        loss1 = self.get_score_loss(pred_scores, golden_scores)
        return loss1, loss2

    def train(self, checkpoint=None):
        logging.info('  开始训练')
        train_data_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=0)
        dev_data_loader = torch.utils.data.DataLoader(dataset=self.dev_dataset, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=0)
        # init
        self.setup()
        # load checkpoint when needed
        checkpoint = self.conf.get("checkpoint", None)
        if checkpoint != '' and checkpoint is not None:
            self.load_model(checkpoint)

        # train
        alpha = 0.5
        while self.epoch < self.epochs:
            logging.info('  epoch: ' + str(self.epoch))
            logging.info('  alpha: ' + str(alpha))
            self.model.train()
            # output lr info
            for lrs in self.optimizer.state_dict()['param_groups']:
                logging.info('  学习率: ' + str(lrs['lr']))
            # training loop
            bar = tqdm(list(enumerate(train_data_loader)))
            for step, input_data in bar:
                inputs = input_data[:-1]
                label = input_data[-1]
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)
                    label = label.cuda()

                pred, pred_scores, golden_scores = self.model(inputs)
                # pred = self.model(inputs)

                # loss calculation
                loss1, loss2 = self.get_loss(pred, pred_scores, golden_scores, label)
                # loss = self.get_loss(pred, None, None, label)
                loss = (1-alpha) * loss1 + alpha * loss2
                # loss = loss1 + loss2
                loss.backward()
                
                # params optimization
                if ((step + 1) % self.accumulate_step) == self.accumulate_step - 1 or step == len(
                        train_data_loader) - 1:
                    self.optimize()

                # calc metric info
                y_pred, y_true = self.get_result(pred, label, input_data)
                precision, recall, f1, metric = self.get_metric(y_pred, y_true)
                bar.set_description(f"step:{step}: pre:{round(precision, 3)} loss1:{round(float(loss1.detach().cpu().numpy()), 3)}, loss2:{round(float(loss2.detach().cpu().numpy()), 3)}")

            # adjust learning rate
            self.scheduler.step()
            alpha *= 0.97
            alpha = max(0.5, alpha)
            # evaluate on dev dataset
            metric, _ = self.eval(self.model, dev_data_loader)
            metric1, _ = self.eval(self.model, dev_data_loader, use_scorer=False)
            metric = max(metric, metric1)
            # model save when achieve best metric
            if metric > self.best_metric:
                self.save_checkpoint(self.model, metric)
                self.best_metric = metric
                self.early_stop = 0
            else:
                self.early_stop += 1
            # early stop
            if self.early_stop > self.early_stop_threshold:
                break
            self.epoch += 1

        logging.info('  训练结束!\n')
        return True

    def eval(self, model, data_loader, mode='dev', ignore_index=-100, use_scorer=True):
        with torch.no_grad():
            model.eval()
            bar = tqdm(list(enumerate(data_loader)))
            ground_truth = []
            pred_label = []
            pred_results = []
            for step, input_data in bar:
                # inputs = tuple(x.to(self.device) for x in input_data[:-1])
                inputs = input_data[:-1]
                label = input_data[-1]
                if self.use_gpu:
                    inputs = tuple(x.cuda() for x in inputs)

                pred = model(inputs, is_predict=True, use_scorer=use_scorer)
                label = label.view(-1).cpu().numpy()
                pred = pred.view(-1).cpu().numpy()
                y_pred, y_true = [], []
                for idx, (p, l) in enumerate(zip(pred, label)):
                    if l == ignore_index:
                        continue
                    y_pred.append(p)
                    y_true.append(l)
                ground_truth += y_true
                pred_label += y_pred
                pred_results.append(y_pred)
            precision, recall, f1, metric = self.get_metric(pred_label, ground_truth)
            logging.info(f"{mode}:{self.epoch}: pre:{round(precision, 3)} rec:{round(recall, 3)} f1:{round(f1, 3)}, use_scorer:{str(use_scorer)}")
        return metric, pred_results

    def test(self, dataset=None, checkpoint=None):
        logging.info('  测试开始')
        if not dataset:
            dataset = self.test_dataset
        # load checkpoint
        if not checkpoint:
            checkpoint = self.best_model_save_path
        self.load_model(checkpoint)
        # load test dataset
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=0)
        # evaluate on test dataset
        metric, pred_results = self.eval(self.model, data_loader, mode='test', use_scorer=False)
        metric, pred_results = self.eval(self.model, data_loader, mode='test')
        logging.info("  测试结果  f1: " + str(metric))
        logging.info("  测试结束!\n")
        return True
