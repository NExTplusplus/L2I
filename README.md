L2I: The baseline method for CCIR 22 competition
====================

This repositary contains the baseline method for CCIR 22 https://www.datafountain.cn/competitions/573

## Dataset
The data used for the competition is from **TAT-QA** ([paper](https://aclanthology.org/2021.acl-long.254.pdf)) ([github repo](https://github.com/NExTplusplus/TAT-QA)) and **TAT-HQA** ([paper](https://aclanthology.org/2022.acl-long.5.pdf)) ([github repo]()). Please refer to the paper for data collection process and baseline method description.

The released training data `tatqa_and_hqa_dataset_train.json` is stored in `dataset_raw`, containing 13,251 factual questions and 6,621 hypothetical questions with ground-truth answers. You can split the data into training and validation set. 

Note that to implement the methods in TAT-QA and TAT-HQA paper, we need to heuristically generate some extra fields from the original data, e.g. answer mapping, deriving operators. The data file containing the generated fields is in `dataset_extra_field`. It is optional to use the file. 

We name the the training and validation data split as `tatqa_and_hqa_field_[train/dev].json` and store in `dataset_extra_field`, which will be processed as the input of the model. 

## Requirements
Please set up the environments according to [TAT-QA github repo](https://github.com/NExTplusplus/TAT-QA), create an environment and download the pre-trained `RoBERTa` model. 

## Data Processing

```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_extra_field --output_dir tag_op/data/roberta --encoder roberta --roberta_model path_to_roberta_model --mode [train/dev]
```
Please fill in the path to roberta_model and select the mode. The processed train/dev data will be stored in `tag_op/data/roberta`. 

## Training

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/trainer.py --data_dir tag_op/data/roberta --save_dir tag_op/model_L2I --batch_size 8 --eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4  --weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --log_per_updates 100 --eps 1e-6  --encoder roberta --test_data_dir tag_op/data/roberta/ --roberta_model path_to_roberta_model 
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/data/roberta --test_data_dir tag_op/data/roberta --save_dir tag_op/model_L2I --eval_batch_size 32 --model_path tag_op/model_L2I --encoder roberta --roberta_model path_to_roberta_model
```

The prediction result for the validation split will be stored in `tag_op/model_L2I/answer_dev.json`

## Result Evaluation

Run `evaluate.py` by specifying the data file with gold answers and the predicted results. This will return the Exact Match and F1 score. 

```bash
python evaluate.py dataset_extra_field/tatqa_and_hqa_field_dev.json tag_op/model_L2I/answer_dev.json 0
```

## Difference with Paper Method
The code in the repo supports joint training of both factual and hypothetical questions. Note that if you want to add the matching block of TAT-HQA, i.e. set the --cross_attn_layer > 0, you probably have to first train on factual questions and fine-tune on hypothetical questions as described in TAT-HQA paper, otherwise the performance of hypothetical questions will have problem. 

## Citation 
```bash
@inproceedings{li2022learning,
  title={Learning to Imagine: Integrating Counterfactual Thinking in Neural Discrete Reasoning},
  author={Li, Moxin and Feng, Fuli and Zhang, Hanwang and He, Xiangnan and Zhu, Fengbin and Chua, Tat-Seng},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={57--69},
  year={2022}
}
@inproceedings{zhu-etal-2021-tat,
    title = "{TAT}-{QA}: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance",
    author = "Zhu, Fengbin  and
      Lei, Wenqiang  and
      Huang, Youcheng  and
      Wang, Chao  and
      Zhang, Shuo  and
      Lv, Jiancheng  and
      Feng, Fuli  and
      Chua, Tat-Seng",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.254",
    doi = "10.18653/v1/2021.acl-long.254",
    pages = "3277--3287"
}
```
## Any Questions? 
Kindly contact us at [limoxin@u.nus.edu](mailto:limoxin@u.nus.edu) for any issue. Thank you!



