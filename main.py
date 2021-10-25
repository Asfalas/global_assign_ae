import argparse
import sys
import json
import logging
import torch

from channels.common.utils import *
from channels.common.const import *
from channels.common.ace05_dataset import ACE05DataHandler
from channels.common.dataset import CommonDataSet
from channels.global_assigner.global_assigner_model import *
from channels.global_assigner.global_assigner_processor import *
from channels.common.processor import CommonModelProcessor

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()


# task setting
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--conf', type=str, default='conf/ace05/ace05.json')
parser.add_argument('--dataset', type=str, default='ace05')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--use_cpu', type=int, default=0)
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--accumulate_step', type=int, default=0)
parser.add_argument('--lr', type=float, default=0)
parser.add_argument('--train_bert', type=int, default=0)
parser.add_argument('--checkpoint', type=str, default=None)

args = parser.parse_args()

def main(args):
    logging.info("-" * 100)
    setup_seed(seed)
    conf = json.load(open(args.conf))
    
    if args.use_cpu:
        conf['use_gpu'] = 0
    else:
        conf['use_gpu'] = torch.cuda.is_available()
    logging.info("  使用gpu? : " + str(bool(conf['use_gpu'])) + '\n')

    conf['pretrain'] = args.pretrain

    # 更新自定义参数
    if args.epochs:
        conf['epochs'] = args.epochs
    if args.batch_size:
        conf['batch_size'] = args.batch_size
    if args.accumulate_step:
        conf['accumulate_step'] = args.accumulate_step
    if args.lr:
        conf["lr"] = args.lr
    if args.train_bert:
        conf["train_bert"] = args.train_bert
    if args.checkpoint:
        conf["checkpoint"] = args.checkpoint
    
    mode = args.mode
    if mode == 'train':
        train_dataset = CommonDataSet(args.dataset, ACE05DataHandler, conf.get("train_path"), conf, debug=args.debug)
        test_dataset = CommonDataSet(args.dataset, ACE05DataHandler, conf.get("test_path"), conf, debug=args.debug)
        dev_dataset = CommonDataSet(args.dataset, ACE05DataHandler, conf.get("dev_path"), conf, debug=args.debug)
    else:
        if args.pretrain:
            conf.update(json.load(open(conf.get('pretrain_best_model_config_path', 'cache/ace05/best_pretrain_config.json'))))
        else:
            conf.update(json.load(open(conf.get('best_model_config_path', "cache/ace05/best_config.json"))))
        train_dataset = None
        dev_dataset = None
        test_dataset = CommonDataSet(args.dataset, ACE05DataHandler, conf.get("test_path"), conf, debug=args.debug)
    
    model = GlobalAssigner(conf)

    # model_processor = CommonModelProcessor(

    model_processor = GlobalAssignerProcessor(
        model,
        train_dataset,
        dev_dataset,
        test_dataset,
        conf
    )

    if mode=='train':
        model_processor.train()
        model_processor.test()
    else:
        model_processor.test()
    return


if __name__ == "__main__":
    logging.info(args)
    main(args)