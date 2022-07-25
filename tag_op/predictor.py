import io, requests, zipfile
import os
import json
import argparse
from datetime import datetime
import options
import torch
import torch.nn as nn
from pprint import pprint
from tools.utils import create_logger, set_environment
from data.tatqa_batch_gen import TaTQATestBatchGen
from data.data_util import OPERATOR_CLASSES_, IF_OPERATOR_CLASSES_
from data.data_util import get_op_1, get_op_2, get_arithmetic_op_index_1, get_arithmetic_op_index_2
from data.data_util import get_op_3, get_arithmetic_op_index_3
from transformers import RobertaModel, BertModel
from tagop.modeling_tagop_L2I import TagopModel
from tools.model import TagopPredictModel
import pandas as pd
from tatqa_metric import TaTQAEmAndF1

parser = argparse.ArgumentParser("Tagop training task.")
options.add_data_args(parser)
options.add_bert_args(parser)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--model_path", type=str)
parser.add_argument("--mode", type=int, default=2)
parser.add_argument("--op_mode", type=int, default=0)
parser.add_argument("--ablation_mode", type=int, default=0)
parser.add_argument("--encoder", type=str, default='roberta')
parser.add_argument("--test_data_dir", type=str, default="tag_op/data/")
parser.add_argument("--finbert_model", type=str, default='cached_models/finbert') # the path to the pretrained finbert model
parser.add_argument("--cross_attn_layer", type=int, default=0) # depth of matching block
parser.add_argument("--ca_with_self", type=int, default=1) # 0 or 1, apply self MHA in matching block?
parser.add_argument("--share_param", type=int, default=1) # 0 or 1, enable parameter sharing in matching block?
parser.add_argument("--result_save_file_name", type=str, default='answer.json') # result file name


args = parser.parse_args()
if args.ablation_mode != 0:
    args.model_path = args.model_path + "_{}_{}".format(args.op_mode, args.ablation_mode)
if args.ablation_mode != 0:
    args.data_dir = args.data_dir + "_{}_{}".format(args.op_mode, args.ablation_mode)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

args.cuda = args.gpu_num > 0

logger = create_logger("Tagop Training", log_file=os.path.join(args.save_dir, args.log_file))

pprint(args)
set_environment(args.cuda)

def main():
    dev_itr = TaTQATestBatchGen(args, data_mode="dev", encoder=args.encoder)
    if args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
    elif args.encoder == 'bert':
        bert_model = BertModel.from_pretrained('bert-large-uncased')
    elif args.encoder == 'finbert':
        bert_model = BertModel.from_pretrained(args.finbert_model)

    if args.ablation_mode == 0:
        operators = [1 for _ in range(10)]
        arithmetic_op_index = [3, 4, 6, 7, 8, 9]
    elif args.ablation_mode == 1:
        operators = get_op_1(args.op_mode)
    elif args.ablation_mode == 2:
        operators = get_op_2(args.op_mode)
    else:
        operators = get_op_3(args.op_mode)

    if args.ablation_mode == 1:
        arithmetic_op_index = get_arithmetic_op_index_1(args.op_mode)
    elif args.ablation_mode == 2:
        arithmetic_op_index = get_arithmetic_op_index_2(args.op_mode)
    else:
        arithmetic_op_index = get_arithmetic_op_index_3(args.op_mode)

    if args.ablation_mode == 0:
        if_operators = IF_OPERATOR_CLASSES_

    network = TagopModel(
        bert=bert_model,
        config=bert_model.config,
        bsz=None,
        operator_classes=len(operators),
        if_operator_classes = len(if_operators),
        scale_classes = 5,
        num_head = 8,
        cross_attn_layer=args.cross_attn_layer,
        ca_with_self=args.ca_with_self,
        share_param=args.share_param,
        #operator_criterion=nn.CrossEntropyLoss(),
        #scale_criterion=nn.CrossEntropyLoss(),
        arithmetic_op_index = arithmetic_op_index,
        op_mode = args.op_mode,
        ablation_mode = args.ablation_mode,
    )
    print("Loading model from", args.model_path)
    state_dict = torch.load(os.path.join(args.model_path,"checkpoint_best.pt"))
    network.load_state_dict(state_dict)

    model = TagopPredictModel(args, network)
    print("*** Evaluation Result ***")
    model.reset()
    model.avg_reset()
    pred_answer = model.predict(dev_itr)
    pred_answer = answer_format(pred_answer)
    answer_file = os.path.join(args.save_dir, args.result_save_file_name.replace('.json', '_dev.json'))
    print('Writing dev answer to', answer_file)
    with open(answer_file, 'w') as f:
        json.dump(pred_answer, f)
    model.get_metrics(logger)


def answer_format(pred_answer):
    pred_answer_format = {}
    for uid, line in pred_answer.items():
        pred_answer_format[uid] = [line["answer"], line["scale"]]
    return pred_answer_format

if __name__ == "__main__":
    main()
