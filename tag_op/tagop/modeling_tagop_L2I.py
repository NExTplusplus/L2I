import torch
import torch.nn as nn
from tatqa_metric import TaTQAEmAndF1
import torch.nn.functional as F
from .util import FFNLayer, CrossAttention, SimpleCrossAttention
from .tools import allennlp as util
from typing import Dict, List, Tuple
from .file_utils import is_scatter_available
import numpy as np
from data.data_util import get_op_1, get_op_2, get_op_3, SCALE, OPERATOR_CLASSES_, IF_OPERATOR_CLASSES_
from tatqa_utils import to_number

np.set_printoptions(threshold=np.inf)
# soft dependency
if is_scatter_available():
    from torch_scatter import scatter
    from torch_scatter import scatter_max


def get_continuous_tag_slots(paragraph_token_tag_prediction):
    tag_slots = []
    span_start = False
    for i in range(1, len(paragraph_token_tag_prediction)):
        if paragraph_token_tag_prediction[i] != 0 and not span_start:
            span_start = True
            start_index = i
        if paragraph_token_tag_prediction[i] == 0 and span_start:
            span_start = False
            tag_slots.append((start_index, i))
    if span_start:
        tag_slots.append((start_index, len(paragraph_token_tag_prediction)))
    return tag_slots


def get_span_tokens_from_paragraph(paragraph_token_tag_prediction, paragraph_tokens) -> List[str]:
    span_tokens = []
    span_start = False
    for i in range(1, min(len(paragraph_tokens) + 1, len(paragraph_token_tag_prediction))):
        if paragraph_token_tag_prediction[i] == 0:
            span_start = False
        if paragraph_token_tag_prediction[i] != 0:
            if not span_start:
                span_tokens.append([paragraph_tokens[i - 1]])
                span_start = True
            else:
                span_tokens[-1] += [paragraph_tokens[i - 1]]
    span_tokens = [" ".join(tokens) for tokens in span_tokens]
    return span_tokens

def get_span_tokens_from_table(table_cell_tag_prediction, table_cell_tokens) -> List[str]:
    span_tokens = []
    for i in range(1, len(table_cell_tag_prediction)):
        if table_cell_tag_prediction[i] != 0:
            span_tokens.append(str(table_cell_tokens[i-1]))
    return span_tokens


def get_single_span_tokens_from_paragraph(paragraph_token_tag_prediction,
                                          paragraph_token_tag_prediction_score,
                                          paragraph_tokens) -> List[str]:
    tag_slots = get_continuous_tag_slots(paragraph_token_tag_prediction)
    best_result = float("-inf")
    best_combine = []
    for tag_slot in tag_slots:
        current_result = np.mean(paragraph_token_tag_prediction_score[tag_slot[0]:tag_slot[1]])
        if current_result > best_result:
            best_result = current_result
            best_combine = tag_slot
    if not best_combine:
        return []
    else:
        return [" ".join(paragraph_tokens[best_combine[0] - 1: best_combine[1] - 1])]

def get_single_span_tokens_from_table(table_cell_tag_prediction,
                                      table_cell_tag_prediction_score,
                                      table_cell_tokens) -> List[str]:
    tagged_cell_index = [i for i in range(len(table_cell_tag_prediction)) if table_cell_tag_prediction[i] != 0]
    if not tagged_cell_index:
        return []
    tagged_cell_tag_prediction_score = \
        [table_cell_tag_prediction_score[i] for i in tagged_cell_index]
    best_result_index = tagged_cell_index[int(np.argmax(tagged_cell_tag_prediction_score))]
    return [str(table_cell_tokens[best_result_index-1])]

def get_numbers_from_reduce_sequence(sequence_reduce_tag_prediction, sequence_numbers):
    return [sequence_numbers[i - 1] for i in
            range(1, min(len(sequence_numbers) + 1, len(sequence_reduce_tag_prediction)))
            if sequence_reduce_tag_prediction[i] != 0 and np.isnan(sequence_numbers[i - 1]) is not True]


def get_numbers_from_table(cell_tag_prediction, table_numbers):
    return [table_numbers[i] for i in range(len(cell_tag_prediction)) if cell_tag_prediction[i] != 0 and \
            np.isnan(table_numbers[i]) is not True]


class TagopModel(nn.Module):
    def __init__(self,
                 bert,
                 config,
                 bsz,
                 operator_classes: int,
                 if_operator_classes:int,
                 scale_classes: int, # 5
                 num_head: int,
                 cross_attn_layer: int,
                 ca_with_self: int,
                 share_param: int,
                 hidden_size: int = None,
                 dropout_prob: float = None,
                 arithmetic_op_index: List = None,
                 op_mode: int = None,
                 ablation_mode: int = None,
                 ):
        super(TagopModel, self).__init__()
        self.encoder = bert
        self.config = config
        self.operator_classes = operator_classes
        self.if_operator_classes = if_operator_classes
        self.scale_classes = scale_classes
        self._metrics = TaTQAEmAndF1(mode=2)
        if hidden_size is None:
            hidden_size = self.config.hidden_size
        if dropout_prob is None:
            dropout_prob = self.config.hidden_dropout_prob
        # if operator predictor
        self.if_operator_predictor = FFNLayer(hidden_size, hidden_size, if_operator_classes, dropout_prob)
        # operator predictor
        self.operator_predictor = FFNLayer(hidden_size, hidden_size, operator_classes, dropout_prob)
        # scale predictor
        self.scale_predictor = FFNLayer(3 * hidden_size, hidden_size, scale_classes, dropout_prob)
        # tag predictor
        self.tag_predictor = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)
        # if tag predictor
        self.if_tag_predictor = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)
        # order predictor
        self.order_predictor = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)
        
        self.share_param = share_param
        self.cross_attn_layer = cross_attn_layer
        print("sharing param?", bool(share_param), "number of CA layers", cross_attn_layer, "with self attn?", bool(ca_with_self))
        if ca_with_self:
            if share_param:
                self.cross_attention = CrossAttention(num_head, hidden_size, dropout_prob)
            else:
                self.cross_attention = nn.ModuleList([CrossAttention(num_head, hidden_size, dropout_prob) for _ in range(self.cross_attn_layer)])
        else:
            if share_param:
                self.cross_attention = SimpleCrossAttention(num_head, hidden_size, dropout_prob)
            else:
                self.cross_attention = nn.ModuleList([SimpleCrossAttention(num_head, hidden_size, dropout_prob) for _ in range(self.cross_attn_layer)])
        
        # criterion for operator/scale loss calculation
        self.if_operator_criterion = nn.CrossEntropyLoss(reduction='none')
        self.operator_criterion = nn.CrossEntropyLoss(reduction='none')
        self.scale_criterion = nn.CrossEntropyLoss(reduction='none')
        # NLLLoss for tag_prediction
        self.NLLLoss = nn.NLLLoss(reduction='none')

        self.config = config
        self.arithmetic_op_index = arithmetic_op_index
        if ablation_mode == 0:
            self.OPERATOR_CLASSES = OPERATOR_CLASSES_
        elif ablation_mode == 1:
            self.OPERATOR_CLASSES = get_op_1(op_mode)
        elif ablation_mode == 2:
            self.OPERATOR_CLASSES = get_op_2(op_mode)
        else:
            self.OPERATOR_CLASSES = get_op_3(op_mode)
        self.IF_OPERATOR_CLASSES = IF_OPERATOR_CLASSES_
        self._metrics = TaTQAEmAndF1()

    """
    :parameter
    input_ids, shape:[bsz, 512] split_tokens' ids, 0 for padded token.
    qtp_attention_mask, shape[bsz, 512], attention mask for question, table and paragraph, excluding assumption. 0 or 1.
    question_if_part_attention_mask, shape[bsz, 512], attention mask for assumption only, 0 or 1
    paragraph_mask, shape[bsz, 512], attention mask for paragraph and question including assumption 0 or 1
    table_mask, shape[bsz, 512], attention mask for table, 0 or 1

    if_tag_labels: [bsz, 512] 1 for tokens used for if tagging and 0 for others
    tag_labels: [bsz, 512] 1 for tokens in the answer and 0 for others
    paragraph_index, shape[bsz, 512] 0 for non-paragraph, non-question tokens and index starting from 1 for paragraph and question tokens, including assumption.
    table_cell_index, shape[bsz, 512] similar to paragraph index.
    operator_labels: [bsz, 10]
    if_operator_labels: [bsz, 9]
    scale_labels: [bsz, 5]
    number_order_labels: [bsz, 2]
    gold_answers: dict, containing gt answer.
    paragraph_tokens: [bsz, text_len], corresponding to paragraph_index
    table_cell_tokens: [bsz, len], corresponding to table_index
    paragraph_numbers: [bsz, text_len], corresponding number extracted from tokens, nan for non-number token
    table_numbers: [bsz, table_size], corresponding number extracted from table cells, nan for non-number cell. Shape is the same as flattened table.
    """
    def forward(self,
                input_ids: torch.LongTensor,  # [cls, q, if, sep, t, sep, p]
                qtp_attention_mask: torch.LongTensor, #[cls, q, 0, sep, 0, sep, p]
                question_if_part_attention_mask: torch.LongTensor, # [cls, 0, if, sep, 0, sep, 0]
                paragraph_mask: torch.LongTensor, # [cls, q, if, sep, 0, sep, p]
                table_mask: torch.LongTensor, # [cls, 0, 0, sep, t, sep, 0]

                token_type_ids: torch.LongTensor,

                if_tag_labels: torch.LongTensor,
                tag_labels: torch.LongTensor, 

                paragraph_index: torch.LongTensor,
                table_cell_index: torch.LongTensor,
                
                operator_labels: torch.LongTensor,
                if_operator_labels: torch.LongTensor,
                scale_labels: torch.LongTensor,
                number_order_labels: torch.LongTensor,
                
                counter_arithmetic_mask: torch.LongTensor, # the input tensor is not used here, re-calculated later
                original_mask: torch.LongTensor, # the input is tensor not used here, re-calculated later
                
                gold_answers: List,
                paragraph_tokens: List[List[str]],
                paragraph_numbers: List[np.ndarray],
                table_cell_tokens: List[List[str]],
                table_cell_numbers: List[np.ndarray],
                
                question_ids: List[str],
                position_ids: torch.LongTensor = None,
                mode=None,
                epoch=None, ) -> Dict[str, torch.Tensor]:
        
        total_attention_mask = qtp_attention_mask + question_if_part_attention_mask
        outputs = self.encoder(
            input_ids,
            attention_mask=total_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)

        batch_size = outputs[0].shape[0]
        device = input_ids.device

        # is a counter arithmetic question? (bsize, )
        arithmetic_mask = (operator_labels == 0) | (operator_labels == 1) | (operator_labels == 2) | (operator_labels == 5)
        arithmetic_mask = ~arithmetic_mask
        counter_mask = question_if_part_attention_mask.sum(-1).bool()
        counter_arithmetic_mask = (arithmetic_mask & counter_mask).int()

        cls_output = outputs[0][:, 0, :]
        operator_prediction = self.operator_predictor(cls_output)
        predicted_operator_class = torch.argmax(operator_prediction, dim=-1)
        if_operator_prediction = self.if_operator_predictor(cls_output) #  bsize if_op_classes
        predicted_if_operator_class = torch.argmax(if_operator_prediction, dim=-1) # bsize
        
        sequence_output = util.replace_masked_values(outputs[0], qtp_attention_mask.unsqueeze(-1), 0)
        if_sequence_output = util.replace_masked_values(outputs[0], question_if_part_attention_mask.unsqueeze(-1), 0)
        
        paragraph_sequence_output = util.replace_masked_values(outputs[0], paragraph_mask.unsqueeze(-1), 0) # paragraph mask contains q and q if part
        table_sequence_output = util.replace_masked_values(outputs[0], table_mask.unsqueeze(-1), 0)
        paragraph_reduce_mean = torch.mean(paragraph_sequence_output, dim=1)
        table_reduce_mean = torch.mean(table_sequence_output, dim=1)
        cls_output = torch.cat((cls_output, table_reduce_mean, paragraph_reduce_mean), dim=-1)
        scale_prediction = self.scale_predictor(cls_output)
        
        if self.share_param:
            for _ in range(self.cross_attn_layer):
                sequence_output, if_sequence_output = self.cross_attention(sequence_output, if_sequence_output, question_if_part_attention_mask, qtp_attention_mask)
        else:
            for layer in self.cross_attention:
                sequence_output, if_sequence_output = layer(sequence_output, if_sequence_output, question_if_part_attention_mask, qtp_attention_mask)
        
        sequence_output = util.replace_masked_values(sequence_output, qtp_attention_mask.unsqueeze(-1), 0)
        if_sequence_output = util.replace_masked_values(if_sequence_output, question_if_part_attention_mask.unsqueeze(-1), 0)
        paragraph_sequence_output = util.replace_masked_values(sequence_output, paragraph_mask.unsqueeze(-1), 0) # 不带if？
        table_sequence_output = util.replace_masked_values(sequence_output, table_mask.unsqueeze(-1), 0)

        concatenated_qtp_if = sequence_output + if_sequence_output
        total_if_tag_prediction = self.if_tag_predictor(concatenated_qtp_if)
        total_if_tag_prediction = util.replace_masked_values(total_if_tag_prediction, total_attention_mask.unsqueeze(-1), 0)
        total_if_tag_prediction = util.masked_log_softmax(total_if_tag_prediction, mask = None)
        total_if_tag_prediction = util.replace_masked_values(total_if_tag_prediction, total_attention_mask.unsqueeze(-1), 0)
        
        total_tag_prediction = self.tag_predictor(concatenated_qtp_if)
        total_tag_prediction = util.replace_masked_values(total_tag_prediction, qtp_attention_mask.unsqueeze(-1), 0)
        total_tag_prediction = util.masked_log_softmax(total_tag_prediction, mask = None)
        total_tag_prediction = util.replace_masked_values(total_tag_prediction, qtp_attention_mask.unsqueeze(-1), 0)
        
        paragraph_mask_only = paragraph_mask - question_if_part_attention_mask # q & p, for predicting the original operands.
        for bsz in range(len(paragraph_mask_only)):
            assert (paragraph_mask_only[bsz] == -1).any() == False
        
        table_reduce_mask = reduce_mean_index(table_mask, table_cell_index)
        paragraph_reduce_mask = reduce_mean_index(paragraph_mask_only, paragraph_index)
        
        table_tag_prediction = util.replace_masked_values(total_tag_prediction, table_mask.unsqueeze(-1), 0)
        paragraph_tag_prediction = util.replace_masked_values(total_tag_prediction, paragraph_mask.unsqueeze(-1), 0)
   
        table_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(table_tag_prediction[:, :, 1], table_sequence_output, table_cell_index) # bsize 512
        table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index) # bsize 512 784
        paragraph_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(paragraph_tag_prediction[:, :, 1], paragraph_sequence_output, paragraph_index)
        paragraph_sequence_reduce_mean_output = reduce_mean_index_vector(paragraph_sequence_output, paragraph_index)

        masked_table_tag_reduce_max_prediction = util.replace_masked_values(table_tag_reduce_max_prediction,
                                                                            table_reduce_mask,
                                                                            -1e+5)
        masked_paragraph_tag_reduce_max_prediction = util.replace_masked_values(paragraph_tag_reduce_max_prediction,
                                                                                paragraph_reduce_mask,
                                                                                -1e+5)
        sorted_table_tag_prediction, sorted_cell_index = torch.sort(masked_table_tag_reduce_max_prediction,
                                                                    dim=-1, descending=True)
        sorted_paragraph_tag_prediction, sorted_paragraph_index = torch.sort(masked_paragraph_tag_reduce_max_prediction,
                                                                             dim=-1, descending=True)
        sorted_table_tag_prediction = sorted_table_tag_prediction[:, :2]
        sorted_cell_index = sorted_cell_index[:, :2]
        sorted_paragraph_tag_prediction = sorted_paragraph_tag_prediction[:, :2]
        sorted_paragraph_index = sorted_paragraph_index[:, :2]
        concat_tag_prediction = torch.cat((sorted_paragraph_tag_prediction, sorted_table_tag_prediction),
                                          dim=1)
        _, sorted_concat_tag_index = torch.sort(concat_tag_prediction, dim=-1, descending=True)
     
        top_2_order_ground_truth = torch.zeros(batch_size).to(device)
        top_2_sequence_output_bw = torch.zeros(batch_size, 2, sequence_output.shape[2]).to(device)
        ground_truth_index = 0
        for bsz in range(batch_size):
            if ("DIVIDE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIVIDE"]) or \
                    ("DIFF" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIFF"]) or \
                    ("CHANGE_RATIO" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["CHANGE_RATIO"]):
                _index = sorted_concat_tag_index[bsz]

                if ("DIVIDE" not in self.OPERATOR_CLASSES or operator_labels[bsz] != self.OPERATOR_CLASSES["DIVIDE"]) and \
                        ("DIFF" not in self.OPERATOR_CLASSES or operator_labels[bsz] != self.OPERATOR_CLASSES["DIFF"]) and \
                        ("CHANGE_RATIO" not in self.OPERATOR_CLASSES or operator_labels[bsz] != self.OPERATOR_CLASSES["CHANGE_RATIO"]):
                    continue
                top_2_order_ground_truth[ground_truth_index] = number_order_labels[bsz]
                
                if _index[0] > 1:
                    top_2_sequence_output_bw[ground_truth_index, 0, :] = table_sequence_reduce_mean_output[bsz,
                                                                         sorted_cell_index[bsz, _index[0] - 2], :]
                else:
                    top_2_sequence_output_bw[ground_truth_index, 0, :] = paragraph_sequence_reduce_mean_output[bsz,
                                                                         sorted_paragraph_index[bsz, _index[0]], :]
                if _index[1] > 1:
                    top_2_sequence_output_bw[ground_truth_index, 1, :] = table_sequence_reduce_mean_output[bsz,
                                                                         sorted_cell_index[bsz, _index[1] - 2], :]
                else:
                    top_2_sequence_output_bw[ground_truth_index, 1, :] = paragraph_sequence_reduce_mean_output[bsz,
                                                                         sorted_paragraph_index[bsz, _index[1]], :]
                ground_truth_index += 1

        top_2_order_prediction_bw = self.order_predictor(
            torch.mean(top_2_sequence_output_bw[:ground_truth_index], dim=1))
        top_2_order_ground_truth = top_2_order_ground_truth[:ground_truth_index]

        output_dict = {}


        # ** Loss Calculation ****
        # if training for both TAT-QA and TAT-HQA, ablate the if operator loss and if tag loss for TAT-QA with original mask.
        operator_prediction_loss = self.operator_criterion(operator_prediction, operator_labels).mean()
        scale_prediction_loss = self.scale_criterion(scale_prediction, scale_labels).mean()
        tag_prediction_loss = self.NLLLoss(total_tag_prediction.transpose(1,2), tag_labels.long()).sum(-1).mean()
        
        if ground_truth_index != 0:
            top_2_order_prediction_bw = util.masked_log_softmax(top_2_order_prediction_bw, mask=None)
            top_2_order_prediction_loss = self.NLLLoss(top_2_order_prediction_bw, top_2_order_ground_truth.long()).mean()
        else:
            top_2_order_prediction_loss = torch.tensor(0, dtype=torch.float).to(device)

        # for counter arithmetic problems only, use counter_arithmetic_mask
        if_operator_prediction_loss = self.if_operator_criterion(if_operator_prediction, if_operator_labels)
        if_operator_prediction_loss = util.replace_masked_values(if_operator_prediction_loss, counter_arithmetic_mask, 0)
        if_tag_prediction_loss = self.NLLLoss(total_if_tag_prediction.transpose(1,2), if_tag_labels.long())
        if_tag_prediction_loss = util.replace_masked_values(if_tag_prediction_loss, counter_arithmetic_mask.unsqueeze(-1), 0).sum(-1)
        if_losses = if_operator_prediction_loss + if_tag_prediction_loss
        if_losses = util.replace_masked_values(if_losses, counter_arithmetic_mask, 0).mean()

        output_dict["top2o_loss"] = top_2_order_prediction_loss.item()
        output_dict["loss"] =  if_losses + tag_prediction_loss + top_2_order_prediction_loss + \
                               scale_prediction_loss + operator_prediction_loss
        return output_dict


    def predict(self,
                input_ids,
                qtp_attention_mask,
                question_if_part_attention_mask,
                paragraph_mask,
                table_mask,
                token_type_ids: torch.LongTensor,

                paragraph_index: torch.LongTensor,
                table_cell_index: torch.LongTensor,
            
                gold_answers,
                paragraph_tokens,
                paragraph_numbers,
                table_cell_tokens,
                table_cell_numbers,
                
                question_ids=None,
                position_ids=None,
                mode=None,
                epoch=None,
                ):
        
        total_attention_mask = qtp_attention_mask + question_if_part_attention_mask
        outputs = self.encoder(
            input_ids,
            attention_mask=total_attention_mask,
            token_type_ids=token_type_ids, # all 0
            position_ids=position_ids)
        
        batch_size = outputs[0].shape[0]
        device = input_ids.device
        
        cls_output = outputs[0][:, 0, :]
        operator_prediction = self.operator_predictor(cls_output)
        predicted_operator_class = torch.argmax(operator_prediction, dim=-1)
        if_operator_prediction = self.if_operator_predictor(cls_output) #  bsize if_op_classes
        predicted_if_operator_class = torch.argmax(if_operator_prediction, dim=-1) # bsize
        
        arithmetic_mask = (predicted_operator_class == 0) | (predicted_operator_class == 1) | (predicted_operator_class == 2) | (predicted_operator_class == 5)
        arithmetic_mask = ~arithmetic_mask
        counter_mask = question_if_part_attention_mask.sum(-1).bool()
        counter_arithmetic_mask = (arithmetic_mask & counter_mask).int()
        
        sequence_output = util.replace_masked_values(outputs[0], qtp_attention_mask.unsqueeze(-1), 0)
        if_sequence_output = util.replace_masked_values(outputs[0], question_if_part_attention_mask.unsqueeze(-1), 0)

        paragraph_sequence_output = util.replace_masked_values(outputs[0], paragraph_mask.unsqueeze(-1), 0)
        table_sequence_output = util.replace_masked_values(outputs[0], table_mask.unsqueeze(-1), 0)
        paragraph_reduce_mean = torch.mean(paragraph_sequence_output, dim=1)
        table_reduce_mean = torch.mean(table_sequence_output, dim=1)
        cls_output = torch.cat((cls_output, table_reduce_mean, paragraph_reduce_mean), dim=-1)

        scale_prediction = self.scale_predictor(cls_output)
        
        if self.share_param:
            for _ in range(self.cross_attn_layer):
                sequence_output, if_sequence_output = self.cross_attention(sequence_output, if_sequence_output, question_if_part_attention_mask, qtp_attention_mask)
        else:
            for layer in self.cross_attention:
                sequence_output, if_sequence_output = layer(sequence_output, if_sequence_output, question_if_part_attention_mask, qtp_attention_mask)
        
        sequence_output = util.replace_masked_values(sequence_output, qtp_attention_mask.unsqueeze(-1), 0)
        if_sequence_output = util.replace_masked_values(if_sequence_output, question_if_part_attention_mask.unsqueeze(-1), 0)
        paragraph_sequence_output = util.replace_masked_values(sequence_output, paragraph_mask.unsqueeze(-1), 0) # q if p
        table_sequence_output = util.replace_masked_values(sequence_output, table_mask.unsqueeze(-1), 0)
        
        concatenated_qtp_if = sequence_output + if_sequence_output
        total_if_tag_prediction = self.if_tag_predictor(concatenated_qtp_if)
        total_if_tag_prediction = util.replace_masked_values(total_if_tag_prediction, total_attention_mask.unsqueeze(-1), 0)
        total_if_tag_prediction = util.masked_log_softmax(total_if_tag_prediction, mask = None)
        total_if_tag_prediction = util.replace_masked_values(total_if_tag_prediction, total_attention_mask.unsqueeze(-1), 0)

        total_tag_prediction = self.tag_predictor(concatenated_qtp_if)
        total_tag_prediction = util.replace_masked_values(total_tag_prediction, qtp_attention_mask.unsqueeze(-1), 0)
        total_tag_prediction = util.masked_log_softmax(total_tag_prediction, mask = None)
        total_tag_prediction = util.replace_masked_values(total_tag_prediction, qtp_attention_mask.unsqueeze(-1), 0)
        
        # question if part top 1 number
        question_if_tag_prediction = util.replace_masked_values(total_if_tag_prediction, question_if_part_attention_mask.unsqueeze(-1), 0)
        question_if_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(question_if_tag_prediction[:, :, 1], if_sequence_output, paragraph_index)
        question_reduce_mask = reduce_mean_index(question_if_part_attention_mask, paragraph_index)
        masked_question_if_tag_reduce_max_prediction = util.replace_masked_values(question_if_tag_reduce_max_prediction,
                                                                                  question_reduce_mask,
                                                                                  -1e+5)
        sorted_question_if_tag_prediction, sorted_question_if_index = torch.sort(masked_question_if_tag_reduce_max_prediction,
                                                                                 dim=-1, descending=True)

        question_top_1_number = np.zeros(batch_size)
        question_if_string = []
        for bsz in range(batch_size):
            #print(sorted_question_if_index[bsz])
            #print(question_if_part_attention_mask[bsz])
            #print(paragraph_numbers[bsz].shape)
            #print(gold_answers[bsz])
            if sorted_question_if_index[bsz, 0] <= paragraph_numbers[bsz].shape[0]:
                question_top_1_number[bsz] = paragraph_numbers[bsz][sorted_question_if_index[bsz, 0] - 1]
                question_if_string.append(paragraph_tokens[bsz][sorted_question_if_index[bsz, 0] - 1])
            else:
                question_top_1_number[bsz] = np.nan
                question_if_string.append([])
        
        table_if_tag_prediction = util.replace_masked_values(total_if_tag_prediction, table_mask.unsqueeze(-1), 0)

        paragraph_mask_only = paragraph_mask - question_if_part_attention_mask # q & p
        #for bsz in range(len(paragraph_mask_only)):
        #    assert (paragraph_mask_only[bsz] == -1).any() == False
        paragraph_if_tag_prediction = util.replace_masked_values(total_if_tag_prediction, paragraph_mask_only.unsqueeze(-1), 0)

        table_if_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(table_if_tag_prediction[:, :, 1], table_sequence_output, table_cell_index) # bsize 512, 是按照index作为下标
        
        paragraph_if_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(paragraph_if_tag_prediction[:, :, 1], paragraph_sequence_output, paragraph_index)
        
        table_reduce_mask = reduce_mean_index(table_mask, table_cell_index)
        paragraph_reduce_mask = reduce_mean_index(paragraph_mask_only, paragraph_index)

        masked_table_if_tag_reduce_max_prediction = util.replace_masked_values(table_if_tag_reduce_max_prediction,
                                                                               table_reduce_mask,
                                                                               -1e+5)
        masked_paragraph_if_tag_reduce_max_prediction = util.replace_masked_values(paragraph_if_tag_reduce_max_prediction,
                                                                                   paragraph_reduce_mask,
                                                                                   -1e+5)

        sorted_table_if_tag_prediction, sorted_cell_if_index = torch.sort(masked_table_if_tag_reduce_max_prediction,
                                                                          dim=-1, descending=True)
                                                                          
        sorted_paragraph_if_tag_prediction, sorted_paragraph_if_index = torch.sort(masked_paragraph_if_tag_reduce_max_prediction,
                                                                                   dim=-1, descending=True)
        sorted_table_if_tag_prediction, sorted_cell_if_index = sorted_table_if_tag_prediction[:, :1], sorted_cell_if_index[:, :1] # 取了top 1
        sorted_paragraph_if_tag_prediction, sorted_paragraph_if_index = sorted_paragraph_if_tag_prediction[:, :1], sorted_paragraph_if_index[:, :1]
        concat_if_tag_prediction = torch.cat((sorted_paragraph_if_tag_prediction, sorted_table_if_tag_prediction),
                                              dim=1)
        _, sorted_concat_if_tag_index = torch.sort(concat_if_tag_prediction, dim=-1, descending=True)

        tp_top_1_number = np.zeros(batch_size)
        tp_if_string = []
        
        for bsz in range(batch_size):
            if sorted_concat_if_tag_index[bsz][0] == 1:
                tp_top_1_number[bsz] = table_cell_numbers[bsz][sorted_cell_if_index[bsz, 0] - 1]
                tp_if_string.append(table_cell_tokens[bsz][sorted_cell_if_index[bsz, 0] - 1])
            else:
                tp_top_1_number[bsz] = paragraph_numbers[bsz][sorted_paragraph_if_index[bsz, 0] - 1]
                tp_if_string.append(paragraph_tokens[bsz][sorted_paragraph_if_index[bsz, 0] - 1])
            if ("NONE" in self.IF_OPERATOR_CLASSES and predicted_if_operator_class[bsz] == self.IF_OPERATOR_CLASSES["NONE"]):
                question_top_1_number[bsz] = tp_top_1_number[bsz] # no need for discrete number replacement.
            elif ("ADD" in self.IF_OPERATOR_CLASSES and predicted_if_operator_class[bsz] == self.IF_OPERATOR_CLASSES["ADD"]):
                question_top_1_number[bsz] += tp_top_1_number[bsz]
            elif ("MINUS" in self.IF_OPERATOR_CLASSES and predicted_if_operator_class[bsz] == self.IF_OPERATOR_CLASSES["MINUS"]):
                question_top_1_number[bsz] = tp_top_1_number[bsz] - question_top_1_number[bsz]
            elif ("MULTIPLY" in self.IF_OPERATOR_CLASSES and predicted_if_operator_class[bsz] == self.IF_OPERATOR_CLASSES["MULTIPLY"]):
                question_top_1_number[bsz] = tp_top_1_number[bsz] * question_top_1_number[bsz]
            elif ("DIVISION" in self.IF_OPERATOR_CLASSES and predicted_if_operator_class[bsz] == self.IF_OPERATOR_CLASSES["DIVISION"]):
                question_top_1_number[bsz] = tp_top_1_number[bsz] / question_top_1_number[bsz]
            elif ("PERCENTAGE_INC" in self.IF_OPERATOR_CLASSES and predicted_if_operator_class[bsz] == self.IF_OPERATOR_CLASSES["PERCENTAGE_INC"]):
                question_top_1_number[bsz] = tp_top_1_number[bsz] * (1 + question_top_1_number[bsz] )
            elif ("PERCENTAGE_DEC" in self.IF_OPERATOR_CLASSES and predicted_if_operator_class[bsz] == self.IF_OPERATOR_CLASSES["PERCENTAGE_DEC"]):
                question_top_1_number[bsz] = tp_top_1_number[bsz] * (1 - question_top_1_number[bsz] )
            # print(question_top_1_number[bsz], tp_top_1_number[bsz], gold_answers[bsz]["gold_if_op"], self.IF_OPERATOR_CLASSES[gold_answers[bsz]["gold_if_op"]], predicted_if_operator_class[bsz])
        
        table_tag_prediction = util.replace_masked_values(total_tag_prediction, table_mask.unsqueeze(-1), 0)
        paragraph_tag_prediction = util.replace_masked_values(total_tag_prediction, paragraph_mask.unsqueeze(-1), 0)

        table_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(table_tag_prediction[:, :, 1], table_sequence_output, table_cell_index) # bsize 512
        table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index) # bsize 512 784 
        paragraph_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(paragraph_tag_prediction[:, :, 1], paragraph_sequence_output, paragraph_index)
        paragraph_sequence_reduce_mean_output = reduce_mean_index_vector(paragraph_sequence_output, paragraph_index)

        masked_table_tag_reduce_max_prediction = util.replace_masked_values(table_tag_reduce_max_prediction,
                                                                            table_reduce_mask,
                                                                            -1e+5)
        masked_paragraph_tag_reduce_max_prediction = util.replace_masked_values(paragraph_tag_reduce_max_prediction,
                                                                                paragraph_reduce_mask,
                                                                                -1e+5)
        sorted_table_tag_prediction, sorted_cell_index = torch.sort(masked_table_tag_reduce_max_prediction,
                                                                    dim=-1, descending=True)
        sorted_paragraph_tag_prediction, sorted_paragraph_index = torch.sort(masked_paragraph_tag_reduce_max_prediction,
                                                                             dim=-1, descending=True)
        
        sorted_table_tag_prediction = sorted_table_tag_prediction[:, :2]
        sorted_cell_index = sorted_cell_index[:, :2]
        sorted_paragraph_tag_prediction = sorted_paragraph_tag_prediction[:, :2]
        sorted_paragraph_index = sorted_paragraph_index[:, :2]
        concat_tag_prediction = torch.cat((sorted_paragraph_tag_prediction, sorted_table_tag_prediction),
                                          dim=1)
        _, sorted_concat_tag_index = torch.sort(concat_tag_prediction, dim=-1, descending=True)
        
        top_2_number = np.zeros((batch_size, 2))
        top_2_sequence_output = torch.zeros(batch_size, 2, sequence_output.shape[2]).to(device)
        number_index = 0
        for bsz in range(batch_size):
            if ("DIVIDE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIVIDE"]) or \
                    ("DIFF" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIFF"]) or \
                    ("CHANGE_RATIO" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["CHANGE_RATIO"]):
                _index = sorted_concat_tag_index[bsz]
                if _index[0] > 1:
                    top_2_number[number_index, 0] = table_cell_numbers[bsz][sorted_cell_index[bsz, _index[0] - 2] - 1]
                    top_2_sequence_output[number_index, 0, :] = table_sequence_reduce_mean_output[bsz,
                                                                sorted_cell_index[bsz, _index[0] - 2], :]
                else:
                    top_2_number[number_index, 0] = paragraph_numbers[bsz][sorted_paragraph_index[bsz, _index[0]] - 1]
                    top_2_sequence_output[number_index, 0, :] = paragraph_sequence_reduce_mean_output[bsz,
                                                                sorted_paragraph_index[bsz, _index[0]], :]
                if _index[1] > 1:
                    top_2_number[number_index, 1] = table_cell_numbers[bsz][sorted_cell_index[bsz, _index[1] - 2] - 1]
                    top_2_sequence_output[number_index, 1, :] = table_sequence_reduce_mean_output[bsz,
                                                                sorted_cell_index[bsz, _index[1] - 2], :]
                else:
                    top_2_number[number_index, 1] = paragraph_numbers[bsz][sorted_paragraph_index[bsz, _index[1]] - 1]
                    top_2_sequence_output[number_index, 1, :] = paragraph_sequence_reduce_mean_output[bsz,
                                                                sorted_paragraph_index[bsz, _index[1]], :]
                number_index += 1
            
        top_2_order_prediction = self.order_predictor(torch.mean(top_2_sequence_output[:number_index], dim=1))
        top_2_number = top_2_number[:number_index]
        
        output_dict = {}
        predicted_tags = torch.argmax(total_tag_prediction, dim = -1).float().cpu().tolist()

        paragraph_tag_prediction_score = paragraph_tag_prediction[:, :, 1]
        paragraph_tag_prediction = torch.argmax(paragraph_tag_prediction, dim=-1).float()
        paragraph_token_tag_prediction = reduce_mean_index(paragraph_tag_prediction, paragraph_index)
        paragraph_token_tag_prediction_score = reduce_max_index(paragraph_tag_prediction_score, paragraph_index)
        paragraph_token_tag_prediction = paragraph_token_tag_prediction.detach().cpu().numpy()
        paragraph_token_tag_prediction_score = paragraph_token_tag_prediction_score.detach().cpu().numpy()
        
        table_tag_prediction_score = table_tag_prediction[:, :, 1]
        table_tag_prediction = torch.argmax(table_tag_prediction, dim=-1).float()
        table_cell_tag_prediction = reduce_mean_index(table_tag_prediction, table_cell_index)
        table_cell_tag_prediction_score = reduce_max_index(table_tag_prediction_score, table_cell_index)
        table_cell_tag_prediction = table_cell_tag_prediction.detach().cpu().numpy()
        table_cell_tag_prediction_score = table_cell_tag_prediction_score.detach().cpu().numpy()

        predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()
        predicted_operator_class = predicted_operator_class.detach().cpu().numpy()
        predicted_if_operator_class = predicted_if_operator_class.detach().cpu().numpy()

        top_2_index = 0
        if number_index != 0:
            top_2_order_prediction = top_2_order_prediction.detach().cpu().numpy()
            top_2_order_prediction = np.argmax(top_2_order_prediction, axis=1)
        for bsz in range(batch_size):
            pred_span = []
            target_fact = None
            predicted_order = None
            current_op = "ignore"
            if "SPAN-TEXT" in self.OPERATOR_CLASSES and \
                    predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SPAN-TEXT"]:
                paragraph_selected_span_tokens = get_single_span_tokens_from_paragraph(
                    paragraph_token_tag_prediction[bsz],
                    paragraph_token_tag_prediction_score[bsz],
                    paragraph_tokens[bsz])

                answer = paragraph_selected_span_tokens
                answer = sorted(answer)
                pred_span += answer
                current_op = "Span-in-text"
            elif "SPAN-TABLE" in self.OPERATOR_CLASSES and \
                 predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SPAN-TABLE"]:
                table_selected_tokens = get_single_span_tokens_from_table(
                    table_cell_tag_prediction[bsz],
                    table_cell_tag_prediction_score[bsz],
                    table_cell_tokens[bsz])
                answer = table_selected_tokens
                answer = sorted(answer)
                pred_span += answer
                current_op = "Cell-in-table"
            elif "MULTI_SPAN" in self.OPERATOR_CLASSES and \
                    predicted_operator_class[bsz] == self.OPERATOR_CLASSES["MULTI_SPAN"]:
                paragraph_selected_span_tokens = \
                    get_span_tokens_from_paragraph(paragraph_token_tag_prediction[bsz], paragraph_tokens[bsz])
                table_selected_tokens = \
                    get_span_tokens_from_table(table_cell_tag_prediction[bsz], table_cell_tokens[bsz])
                        
                answer = paragraph_selected_span_tokens + table_selected_tokens
                answer = sorted(answer)
                pred_span += answer
                current_op = "Spans"
            elif "COUNT" in self.OPERATOR_CLASSES and \
                    predicted_operator_class[bsz] == self.OPERATOR_CLASSES["COUNT"]:
                paragraph_selected_tokens = \
                    get_span_tokens_from_paragraph(paragraph_token_tag_prediction[bsz], paragraph_tokens[bsz])
                table_selected_tokens = \
                    get_span_tokens_from_table(table_cell_tag_prediction[bsz], table_cell_tokens[bsz])
                        
                answer = len(paragraph_selected_tokens) + len(table_selected_tokens)
                pred_span += sorted(paragraph_selected_tokens + table_selected_tokens)
                current_op = "Count"
            else:
                if ("SUM" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SUM"]) \
                        or ("TIMES" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["TIMES"]) \
                        or ("AVERAGE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["AVERAGE"]):
                    paragraph_selected_numbers = \
                        get_numbers_from_reduce_sequence(paragraph_token_tag_prediction[bsz],
                                                         paragraph_numbers[bsz])
                    table_selected_numbers = \
                        get_numbers_from_reduce_sequence(table_cell_tag_prediction[bsz], table_cell_numbers[bsz])
                    selected_numbers = paragraph_selected_numbers + table_selected_numbers
                    
                    if counter_arithmetic_mask[bsz]: # must be a counter arithmetic question, must change number. if do not change number, set false
                        new_number = question_top_1_number[bsz]
                        to_cover_number = tp_top_1_number[bsz]
                        target_fact = to_cover_number
                        for i, num in enumerate(selected_numbers):
                            if num == to_cover_number:
                                selected_numbers[i] = new_number
                                break
                                
                    pred_span += sorted(selected_numbers)
                    if not selected_numbers:
                        answer = ""
                    elif "SUM" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES[
                        "SUM"]:
                        answer = np.around(np.sum(selected_numbers), 4)
                        #print("SUM", answer, selected_numbers)
                        current_op = "Sum"
                    elif "TIMES" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES[
                        "TIMES"]:
                        answer = np.around(np.prod(selected_numbers), 4)
                        #print("TIMES", answer, selected_numbers)
                        current_op = "Multiplication"
                    elif "AVERAGE" in self.OPERATOR_CLASSES:
                        answer = np.around(np.mean(selected_numbers), 4)
                        #print("AVERAGE", answer, selected_numbers)
                        current_op = "Average"
                else:
                    if top_2_number.size <= 0:
                        answer = ""
                    if top_2_number.size > 0:
                        operand_one = top_2_number[top_2_index, 0]
                        operand_two = top_2_number[top_2_index, 1]
                        # print(operand_one, operand_two, new_number, to_cover_number)
                        predicted_order = top_2_order_prediction[top_2_index]
                        if np.isnan(operand_one) or np.isnan(operand_two):
                            answer = ""
                        else:
                            if counter_arithmetic_mask[bsz]: # must be a counter arithmetic question, must change number. if do not change number, set false
                                new_number = question_top_1_number[bsz]
                                to_cover_number = tp_top_1_number[bsz]
                                target_fact = to_cover_number
                                if operand_one == to_cover_number:
                                    operand_one = new_number
                                elif operand_two == to_cover_number:
                                    operand_two = new_number
                                if "DIFF" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == \
                                    self.OPERATOR_CLASSES["DIFF"]:
                                    current_op = "Difference"
                                    if predicted_order == 0:
                                        answer = np.around(operand_one - operand_two, 4)
                                    else:
                                        answer = np.around(operand_two - operand_one, 4)
                                elif "DIVIDE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == \
                                    self.OPERATOR_CLASSES["DIVIDE"]:
                                    current_op = "Division"
                                    if "SWAP_MIN_NUM" in self.IF_OPERATOR_CLASSES and predicted_if_operator_class[bsz] == self.IF_OPERATOR_CLASSES["SWAP_MIN_NUM"]:
                                        operand_one = operand_one - to_cover_number + new_number
                                        operand_two = operand_two - to_cover_number + new_number
                                    if predicted_order == 0:
                                        answer = np.around(operand_one / operand_two, 4)
                                    else:
                                        answer = np.around(operand_two / operand_one, 4)
                                    if SCALE[int(predicted_scale_class[bsz])] == "percent":
                                        answer = answer * 100
                                elif "CHANGE_RATIO" in self.OPERATOR_CLASSES:
                                    current_op = "Change ratio"
                                    if predicted_order == 0:
                                        answer = np.around(operand_one / operand_two - 1, 4)
                                    else:
                                        answer = np.around(operand_two / operand_one - 1, 4)
                                    if SCALE[int(predicted_scale_class[bsz])] == "percent":
                                        answer = answer * 100
                            else:
                                if "DIFF" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIFF"]:
                                    current_op = "Difference"
                                    if predicted_order == 0:
                                        answer = np.around(operand_one - operand_two, 4)
                                    else:
                                        answer = np.around(operand_two - operand_one, 4)
                                elif "DIVIDE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIVIDE"]:
                                    current_op = "Division"
                                    if predicted_order == 0:
                                        answer = np.around(operand_one / operand_two, 4)
                                    else:
                                        answer = np.around(operand_two / operand_one, 4)
                                    if SCALE[int(predicted_scale_class[bsz])] == "percent":
                                        answer = answer * 100
                                elif "CHANGE_RATIO" in self.OPERATOR_CLASSES:
                                    current_op = "Change ratio"
                                    if predicted_order == 0:
                                        answer = np.around(operand_one / operand_two - 1, 4)
                                    else:
                                        answer = np.around(operand_two / operand_one - 1, 4)
                                    if SCALE[int(predicted_scale_class[bsz])] == "percent":
                                        answer = answer * 100
                        pred_span += [operand_one, operand_two]
                        top_2_index += 1

            output_dict[question_ids[bsz]] = {"answer": answer, "scale": SCALE[int(predicted_scale_class[bsz])]}
            
            if predicted_operator_class[bsz] in self.arithmetic_op_index:
                predict_type = "arithmetic"
            else:
                predict_type = ""
            em, f1 = self._metrics({**gold_answers[bsz], "uid": question_ids[bsz]}, answer, predict_type,
                          SCALE[int(predicted_scale_class[bsz])])
            output_dict[question_ids[bsz]]["em"] = em
            output_dict[question_ids[bsz]]["f1"] = f1
            
        return output_dict


    def reset(self):
        self._metrics.reset()

    def set_metrics_mdoe(self, mode):
        self._metrics = TaTQAEmAndF1(mode=mode)

    def get_metrics(self, logger=None, reset: bool = False) -> Dict[str, float]:
        detail_em, detail_f1 = self._metrics.get_detail_metric()
        raw_detail = self._metrics.get_raw_pivot_table()
        exact_match, f1_score, scale_score = self._metrics.get_overall_metric(reset)
        '''
        print(f"raw matrix:{raw_detail}\r\n")
        print(f"detail em:{detail_em}\r\n")
        print(f"detail f1:{detail_f1}\r\n")
        print(f"global em:{exact_match}\r\n")
        print(f"global f1:{f1_score}\r\n")
        print(f"global scale:{scale_score}\r\n")
        print(f"global op:{op_score}\r\n")
        '''
        if logger is not None:
            logger.info(f"raw matrix:{raw_detail}\r\n")
            logger.info(f"detail em:{detail_em}\r\n")
            logger.info(f"detail f1:{detail_f1}\r\n")
            logger.info(f"global em:{exact_match}\r\n")
            logger.info(f"global f1:{f1_score}\r\n")
            logger.info(f"global scale:{scale_score}\r\n")
            #logger.info(f"global op:{op_score}\r\n")
            #logger.info(f"global order:{order_score}\r\n")
        return {'em': exact_match, 'f1': f1_score, "scale": scale_score}

    def get_df(self):
        raws = self._metrics.get_raw()
        detail_em, detail_f1 = self._metrics.get_detail_metric()
        raw_detail = self._metrics.get_raw_pivot_table()
        return detail_em, detail_f1, raws, raw_detail


def calc_KLD(p, q):
    kld = p * torch.log(p / q) + q * torch.log(q / p)
    kld = kld / 2
    kld = kld.sum(-1).mean(-1)
    return kld

### Beginning of everything related to segmented tensors ###


class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index
        Args:
            indices (:obj:`torch.LongTensor`, same shape as a `values` Tensor to which the indices refer):
                Tensor containing the indices.
            num_segments (:obj:`torch.LongTensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor must have the same
                number of segments (although many segments can be empty).
            batch_dims (:obj:`int`, `optional`, defaults to 0):
                The number of batch dimensions. The first `batch_dims` dimensions of a SegmentedTensor are treated as
                batch dimensions. Segments in different batch elements are always distinct even if they have the same
                index.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments, device=indices.device)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return self.indices.size()[: self.batch_dims]  # returns a torch.Size object


class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segments` * `inner_index.num_segments`
        Args:
            outer_index (:obj:`IndexMap`):
                IndexMap.
            inner_index (:obj:`IndexMap`):
                IndexMap, must have the same shape as `outer_index`.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super(ProductIndexMap, self).__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=(index.indices // self.inner_index.num_segments).type(torch.float).floor().type(torch.long),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims,
        )

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments)
                .type(torch.float)
                .floor()
                .type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )


def reduce_mean_vector(values, index, name="segmented_reduce_vector_mean"):
    return _segment_reduce_vector(values, index, "mean", name)


def reduce_mean(values, index, name="segmented_reduce_mean"):
    """
    Averages a tensor over its segments.
    Outputs 0 for empty segments.
    This operations computes the mean over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a mean of
          vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the mean must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used
    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (:obj:`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "mean", name)


def reduce_mean_index_vector(values, index, max_length=512, name="index_reduce_mean"):
    return _index_reduce_vector(values, index, max_length, "mean", name)


def reduce_mean_index(values, index, max_length=512, name="index_reduce_mean"):
    return _index_reduce(values, index, max_length, "mean", name)


def reduce_max_index(values, index, max_length=512, name="index_reduce_max"):
    return _index_reduce_max(values, index, max_length, name)


def reduce_max_index_get_vector(values_for_reduce, values_for_reference, index,
                                max_length=512, name="index_reduce_get_vector"):
    return _index_reduce_max_get_vector(values_for_reduce, values_for_reference, index, max_length, name)


def flatten(index, name="segmented_flatten"):
    """
    Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map. This operation
    relabels the segments to keep batch elements distinct. The k-th batch element will have indices shifted by
    `num_segments` * (k - 1). The result is a tensor with `num_segments` multiplied by the number of elements in the
    batch.
    Args:
        index (:obj:`IndexMap`):
            IndexMap to flatten.
        name (:obj:`str`, `optional`, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): The flattened IndexMap.
    """
    # first, get batch_size as scalar tensor
    batch_size = torch.prod(torch.tensor(list(index.batch_shape())))
    # next, create offset as 1-D tensor of length batch_size,
    # and multiply element-wise by num segments (to offset different elements in the batch) e.g. if batch size is 2: [0, 64]
    offset = torch.arange(start=0, end=batch_size, device=index.num_segments.device) * index.num_segments
    offset = offset.view(index.batch_shape())
    for _ in range(index.batch_dims, len(index.indices.size())):  # typically range(1,2)
        offset = offset.unsqueeze(-1)

    indices = offset + index.indices
    return IndexMap(indices=indices.view(-1), num_segments=index.num_segments * batch_size, batch_dims=0)


def flatten_index(index, max_length=512, name="index_flatten"):
    batch_size = index.shape[0]
    offset = torch.arange(start=0, end=batch_size, device=index.device) * max_length
    offset = offset.view(batch_size, 1)
    return (index + offset).view(-1), batch_size * max_length


def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).
    Args:
        batch_shape (:obj:`torch.Size`):
            Batch shape
        num_segments (:obj:`int`):
            Number of segments
        name (:obj:`str`, `optional`, defaults to 'range_index_map'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = torch.as_tensor(
        batch_shape, dtype=torch.long
    )  # create a rank 1 tensor vector containing batch_shape (e.g. [2])
    assert len(batch_shape.size()) == 1
    num_segments = torch.as_tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    assert len(num_segments.size()) == 0

    indices = torch.arange(
        start=0, end=num_segments, device=num_segments.device
    )  # create a rank 1 vector with num_segments elements
    new_tensor = torch.cat(
        [torch.ones_like(batch_shape, dtype=torch.long, device=num_segments.device), num_segments.unsqueeze(dim=0)],
        dim=0,
    )
    # new_tensor is just a vector of [1 64] for example (assuming only 1 batch dimension)
    new_shape = [int(x) for x in new_tensor.tolist()]
    indices = indices.view(new_shape)

    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    indices = indices.repeat(multiples.tolist())
    # equivalent (in Numpy:)
    # indices = torch.as_tensor(np.tile(indices.numpy(), multiples.tolist()))

    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.size())[0])


def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.
    Args:
        values (:obj:`torch.Tensor`):
            Tensor with segment values.
        index (:obj:`IndexMap`):
            IndexMap.
        segment_reduce_fn (:obj:`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (:obj:`str`):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()):]  # torch.Size object
    flattened_shape = torch.cat(
        [torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0
    )
    # changed "view" by "reshape" in the following line
    flat_values = values.reshape(flattened_shape.tolist())

    segment_means = scatter(
        src=flat_values,
        index=flat_index.indices.type(torch.long),
        dim=0,
        dim_size=flat_index.num_segments,
        reduce=segment_reduce_fn,
    )

    # Unflatten the values.
    new_shape = torch.cat(
        [
            torch.as_tensor(index.batch_shape(), dtype=torch.long),
            torch.as_tensor([index.num_segments], dtype=torch.long),
            torch.as_tensor(vector_shape, dtype=torch.long),
        ],
        dim=0,
    )

    output_values = segment_means.view(new_shape.tolist())
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def _segment_reduce_vector(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.
    Args:
        values (:obj:`torch.Tensor`):
            Tensor with segment values.
        index (:obj:`IndexMap`):
            IndexMap.
        segment_reduce_fn (:obj:`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (:obj:`str`):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()):]  # torch.Size object
    bsz = values.shape[0]
    seq_len = values.shape[1]
    hidden_size = values.shape[2]
    flat_values = values.reshape(bsz * seq_len, hidden_size)
    segment_means = scatter(
        src=flat_values,
        index=flat_index.indices.type(torch.long),
        dim=0,
        dim_size=flat_index.num_segments,
        reduce=segment_reduce_fn,
    )
    output_values = segment_means.view(bsz, -1, hidden_size)
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def _index_reduce(values, index, max_length, index_reduce_fn, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values.shape[0]
    seq_len = values.shape[1]
    flat_values = values.reshape(bsz * seq_len)
    index_means = scatter(
        src=flat_values,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
        reduce=index_reduce_fn,
    )
    output_values = index_means.view(bsz, -1)
    return output_values


def _index_reduce_max(values, index, max_length, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values.shape[0]
    seq_len = values.shape[1]
    flat_values = values.reshape(bsz * seq_len)
    index_max, _ = scatter_max(
        src=flat_values,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
    )
    output_values = index_max.view(bsz, -1)
    return output_values


def _index_reduce_max_get_vector(values_for_reduce, values_for_reference, index, max_length, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values_for_reduce.shape[0]
    seq_len = values_for_reference.shape[1]
    flat_values_for_reduce = values_for_reduce.reshape(bsz * seq_len)
    flat_values_for_reference = values_for_reference.reshape(bsz * seq_len, -1) # bsize*seqlen 784
    reduce_values, reduce_index = scatter_max(
        src=flat_values_for_reduce,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
    )
    reduce_index[reduce_index == -1] = flat_values_for_reference.shape[0]
    reduce_values = reduce_values.view(bsz, -1)
    flat_values_for_reference = torch.cat(
        (flat_values_for_reference, torch.zeros(1, flat_values_for_reference.shape[1]).to(values_for_reduce.device)),
        dim=0)
    flat_values_for_reference = torch.index_select(flat_values_for_reference, dim=0, index=reduce_index)
    flat_values_for_reference = flat_values_for_reference.view(bsz, reduce_values.shape[1], -1)
    return reduce_values, flat_values_for_reference # bsize seq_len, 784.


def _index_reduce_vector(values, index, max_length, index_reduce_fn, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values.shape[0]
    seq_len = values.shape[1]
    hidden_size = values.shape[2]
    flat_values = values.reshape(bsz * seq_len, hidden_size)
    index_means = scatter(
        src=flat_values,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
        reduce=index_reduce_fn,
    )
    output_values = index_means.view(bsz, -1, hidden_size)
    return output_values
