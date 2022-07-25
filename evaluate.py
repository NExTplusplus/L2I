"""Official evaluation script for KCT version 1.0.

Some code here has been copied from:
   https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
with some modifications.
"""
import argparse
import json
import re
import string
import sys
from typing import Set, Union, Any, Dict, Tuple, List

import numpy as np
from scipy.optimize import linear_sum_assignment

OPTS = None


def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for KAPT version 1.0.')
    parser.add_argument('data_file', metavar='gold_answer.json', help='Input data json file.')
    parser.add_argument('pred_file', metavar='pred_answer.json', help='Model predictions in json.')
    parser.add_argument('evalStrategy', metavar='a =0,b =1',help='evalStrategy', default=0)

    if len(sys.argv) != 4:
        print('argument has error,' + str(len(sys.argv)) + ' not equal 3')
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def scale_to_num(scale):
    scale = scale.lower()
    num = 1
    if 'hundred' in scale:  # hundred
        num = 100
    elif 'thousand' in scale:  # thousand
        num = 1000
    elif 'million' in scale:  # million
        num = 1000000
    elif 'billion' in scale:  # billion
        num = 1000000000
    elif 'percent' in scale:  # percent
        num = 0.01
    return num

def extract_one_num_from_str(s):
    s = _clean_num(s)
    r_num = r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)"
    groups = re.findall(r_num, s)
    if len(groups) == 0:
        return None
    num = groups[0][0]
    if num == '':
        return None
    if '.' in num:
        return float(num)
    return int(num)

EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"
def _clean_num(text:str):
    return "".join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])


def is_number(text: str) -> bool:
    try:
        words = " ".join([_clean_num(w) for w in text.split()]).split()
        if len(words) == 0:
            """1023 or 1 million"""
            return False
        num = float(words[0])
        if np.isnan(num):
            return False
        if len(words) >= 2:
            if scale_to_num(words[1]) == 1:
                return False
        return True
    except ValueError:
        return False
    # except AttributeError:
    #     return False

def negative_num_handle(x):
    """
    :param x:  transform (134) -> -134
    :return:
    """
    all = re.findall('(\([\d.\s]+\))', x.strip())
    if len(all) > 0:
        return -1
    return 1

def percent_num_handle(x):
    """
    :param x:  transform 12% -> 12/100
    :return:
    """
    all = re.findall('([\d.\s]+%)', x.strip())
    if len(all) > 0:
        return 0.01
    return 1

def word_scale_handle(x):
    """
    :param x: 1 million = 1,000,000
    :return:
    """
    iter = re.finditer('([\d.]+\s?[a-zA-Z]+)', x)
    for one in iter:
        text = one.group(0).lower()
        scale_val = scale_to_num(text)
        return scale_val
    return 1

def to_number(text:str) -> float:
    num = extract_one_num_from_str(text)
    scale_val = word_scale_handle(text)
    negative_flag = negative_num_handle(text)
    percent_flag = percent_num_handle(text)
    if num is not None:
        return round(num * scale_val * negative_flag * percent_flag, 4)
    return None

def remove_articles(text: str) -> str:
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def white_space_fix(text: str) -> str:
    return ' '.join(text.split())

EXCLUDE = set(string.punctuation)
def remove_punc(text: str) -> str:
    if not is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text

def lower(text: str) -> str:
    return text.lower()

def tokenize(text: str) -> List[str]:
    return re.split(" ", text)


def normalize_number(text: str) -> str:
    if is_number(text):
        return str(to_number(text))
    else:
        return text

def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [white_space_fix(remove_articles(normalize_number(remove_punc(lower(token)))))
             for token in tokenize(text)]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized


STRIPPED_CHARACTERS = string.punctuation + ''.join([u"‘", u"’", u"´", u"`", "_"])
def ws_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip().lower()
    if not text:
        return []
    text = white_space_fix(text)
    tokens = text.split()
    tokens = [token.strip(STRIPPED_CHARACTERS) for token in tokens]
    return tokens


def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            # if _match_numbers_if_present(gold_item, pred_item): no need to match number in tatqa
            scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(predicted: Union[str, List[str], Tuple[str, ...]],
                gold: Union[str, List[str], Tuple[str, ...]]) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def extract_gold_answers(qa_annotation):
    '''
    span
    multi-span
    arithmetic (+ - * /)
    count
    date
    other
    gold answers is a list of list, each item in gold answers is a valid answer
    '''
    answer_type, scale = qa_annotation["answer_type"], qa_annotation['scale']
    answer_content = qa_annotation['answer']
    gold_answers = []
    if answer_type in ['multi-span', 'span']: # list
        assert isinstance(answer_content, list), answer_content
        gold_answers = answer_content # multi-span
    elif answer_type in ["arithmetic"]:
        gold_answers.append(str(answer_content))
    elif answer_type in ['count']:
        gold_answers.append(str(int(answer_content)))
    else:
        gold_answers.append(str(answer_content))
    return answer_type, gold_answers, scale


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
    scores_for_ground_truths = []
    for pred in predictions:
        for ground_truth in ground_truths:
            score = metric_fn(pred, ground_truth)
            scores_for_ground_truths.append(score)
    if len(scores_for_ground_truths) == 0:
        return 0, 0
    return max(scores_for_ground_truths)



def get_answer_str(answers: list, scale: str):
    """
    :param ans_type:  span, multi-span, arithmetic, count
    :param ans_list:
    :param scale: "", thousand, million, billion, percent
    :param mode:
    :return:

    """
    sorted_ans = sorted(answers)
    ans_temp = []
    for ans in sorted_ans:
        ans_str = str(ans)
        if is_number(ans_str):
            ans_num = to_number(ans_str)
            if ans_num is None:
                if scale:
                    ans_str = ans_str + " " + str(scale)
            else:
                if '%' in ans_str: #  has been handled the answer itself is a percentage
                    ans_str = '%.4f' % ans_num
                else:
                    ans_str = '%.4f' % (round(ans_num, 2) * scale_to_num(scale))
        else:
            if scale:
                ans_str = ans_str + " " + str(scale)
        ans_temp.append(ans_str)
    return [" ".join(ans_temp)]

# handle percentage
def add_percent_pred(prediction_strings, pred_scale, pred):
    """
    to solve [pred = 0.2342] <>   [ans = 23.42 and scale == 'percent']

    :param prediction_strings:
    :param gold_ans_type:
    :param gold_scale:
    :param pred:
    :return:
    """
    if len(pred) > 1:
        return prediction_strings
    pred_str = str(pred[0])
    if pred_str is None:
        return prediction_strings
    if not pred_scale and '%' not in pred_str and is_number(pred_str): # mode only or no pred_scale num only
        pred_str = to_number(pred_str)
        if pred_str is None:
            return prediction_strings
        prediction_strings.append('%.4f' % pred_str)
    return prediction_strings


class TaTQAEmAndF1(object):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._scale_em = 0.0
        self._op_em = 0.0
        self.op_correct_count = {"Span-in-text": 0, "Cell-in-table": 0, "Spans": 0, "Sum": 0, "Count": 0, "Average": 0,
                         "Multiplication": 0, "Division": 0, "Difference": 0, "Change ratio": 0, "ignore":0}
        # self.op_total_count = {"Span-in-text": 0, "Cell-in-table": 0, "Spans": 0, "Sum": 0, "Count": 0, "Average": 0,
        #                  "Multiplication": 0, "Division": 0, "Difference": 0, "Change ratio": 0, "ignore":0}
        self.scale_correct_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        self.scale_total_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        self._count = 0
        self._details = []

    def __call__(self,
                 ground_truth: dict,
                 prediction: Union[str, List],
                 pred_scale="",
                 pred_span = None,
                 gold_span = None,
                 pred_op=None,
                 gold_op=None):  # type: ignore
        """
        :param ground_truth:
        :param prediction:
        :param pred_scale:
        :param pred_span:
        :param gold_span:
        :param pred_op:
        :param gold_op:
        :return:
        """
        # if pred_op is not None:
        #     if pred_op == gold_op:
        #         self.op_correct_count[pred_op] += 1
        #         self._op_em += 1
        #     self.op_total_count[gold_op] += 1

        if pred_scale == ground_truth["scale"]:
            self.scale_correct_count[pred_scale] += 1

        self.scale_total_count[ground_truth["scale"]] += 1
        if not prediction:
            exact_match = 0
            f1_score = 0
            span_exact_match = 0
            span_f1_score = 0
        else:
            gold_type, gold_answer, gold_scale = extract_gold_answers(ground_truth)
            if not gold_answer:
                exact_match = 0
                f1_score = 0
                span_exact_match = 0
                span_f1_score = 0
            else:
                ground_truth_answer_strings = get_answer_str(gold_answer, gold_scale)

                if gold_scale == pred_scale:
                    self._scale_em += 1
                prediction = prediction if isinstance(prediction, list) else [prediction]
                prediction_strings = get_answer_str(prediction, pred_scale)
                prediction_strings = add_percent_pred(prediction_strings, pred_scale, prediction)
                exact_match, f1_score = metric_max_over_ground_truths(
                        get_metrics,
                        prediction_strings,
                        ground_truth_answer_strings
                )
                if gold_type in ['arithmetic', 'count']:
                    """if gold type equals with arithmetic and count, set the f1_score == exact_match"""
                    f1_score = exact_match
                if not pred_span:
                    span_exact_match = 0
                    span_f1_score = 0
                else:
                    pred_span_strings = get_answer_str(pred_span, "")
                    gold_span_strings = get_answer_str(gold_span, "")
                    span_exact_match, span_f1_score = metric_max_over_ground_truths(
                        get_metrics,
                        pred_span_strings,
                        gold_span_strings,
                    )

        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1
        it = {**ground_truth,
              **{"pred":prediction,
                 "pred_scale":pred_scale,
                 "em":exact_match,
                 "f1":f1_score,
                 "pred_span":pred_span,
                 "gold_span":gold_span,
                 "span_em":span_exact_match,
                 "span_f1":span_f1_score}}
        self._details.append(it)

    def get_overall_metric(self, reset: bool = False) -> Tuple[float, float, float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        scale_score = self._scale_em / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score, scale_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._scale_em = 0.0
        self._op_em = 0.0
        self._count = 0
        self._details = []
        # self.op_correct_count = {"Span-in-text": 0, "Cell-in-table": 0, "Spans": 0, "Sum": 0, "Count": 0, "Average": 0,
        #                          "Multiplication": 0, "Division": 0, "Difference": 0, "Change ratio": 0, "ignore":0}
        # self.op_total_count = {"Span-in-text": 0, "Cell-in-table": 0, "Spans": 0, "Sum": 0, "Count": 0, "Average": 0,
        #                        "Multiplication": 0, "Division": 0, "Difference": 0, "Change ratio": 0, "ignore":0}
        self.scale_correct_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        self.scale_total_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}

    def __str__(self):
        return f"TaTQAEmAndF1(em={self._total_em}, f1={self._total_f1}, count={self._count})"


def evaluate_json(golden_answers: Dict[str, Any], predicted_answers: Dict[str, Any]) -> Tuple[float, float]:

    em_and_f1 = TaTQAEmAndF1()
    for qas in golden_answers:
        for qa in qas["questions"]:
            query_id = qa["uid"]
            pred_answer, pred_scale = None, None
            if query_id in predicted_answers:
                pred_answer, pred_scale = predicted_answers[query_id]
            em_and_f1(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)

    global_em, global_f1, global_scale = em_and_f1.get_overall_metric()
    # print("----")
    # print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    # print("F1 score {0:.2f}".format(global_f1 * 100))
    # print("Scale score {0:.2f}".format(global_scale * 100))
    # print("Opera score {0:.2f}".format(global_scale * 100))
    # # print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    # print("----")
    return global_em, global_f1



def judge(standardResultFile, userCommitFile, evalStrategy=0):
    """
	评估策略：样本不均衡，计算F1得分
	standardResultFile  :  标准结果文件路径, level字段 0-A榜标签 1-B榜标签 2-A/B榜都可使用
	userCommitFile:  用户提交结果路径
	evalStrategy 0/1:  0表示计算A榜成绩，1表示计算B榜成绩
	"""
    golden_answers = json.load(open(standardResultFile, encoding='utf-8'))
    predicted_answers = json.load(open(userCommitFile, encoding='utf-8'))
    return evaluate_json(golden_answers, predicted_answers)


if __name__ == '__main__':
    OPTS = parse_args()
    standardResultFile = OPTS.data_file
    userCommitFile = OPTS.pred_file

    result = judge(standardResultFile, userCommitFile, evalStrategy=0)
    print(result[0], end='')
    print(';', end='')
    print(result[1], end='')
    # print(';', end='')
    # print(result[2], end='')
    # print(';', end='')
    # print(result[3], end='')
