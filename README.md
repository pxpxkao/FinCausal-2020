# NTUNLPL at FinCausal 2020

## Introduction
Source code of **NTUNLPL at FinCausal 2020, Task 2: Improving Causality DetectionUsing Viterbi Decoder**.
This approach is ranked in the 1st place in the official run.

#### Blind Test Result

| Ranking        | F1-Score     | Recall       | Precision    | Exact Match  |
|:-------------- | ------------ |:------------ |:------------ |:------------ |
| 1-Our Approach | 0.947154(1)  | 0.947012(1)  | 0.947870(1)  | 0.824451(1)  |
| 2              | 0.946622 (2) | 0.946624 (2) | 0.946659 (2) | 0.736677 (2) |
| 3              | 0.837098 (3) | 0.836262 (3) | 0.839150 (3) | 0.703762 (4) |

## Quick Start
Simple demo for our training and testing procedures.
```shell
bash run.sh
```
We provide some sample training and testing data(`train.csv`, `test.csv`, `test_gold.csv`) in **data** folder just to clarify our input format.
### Details in run.sh
#### preprocess.py
* **Input** : `data/train.csv` & `data/test.csv`
* **Output** : `data/train.txt` & `data/test.txt`
* Use [Stanza](https://stanfordnlp.github.io/stanza/) toolkit to tokenize each sentence and generate the part-of-speech (POS) tag for each token.

#### run.py
* --add_pos : Whether to use POS tag features.
* --do_train : Whether to run training.
* --do_predict : Whether to run predictions on the test set.

#### submission.py
* **Input** : `$OUTPUT_DIR/test_predictions.txt` & `data/test.csv`
* **Output** : `pred.csv`
* Post-process of BIO tagging scheme.

### Evaluation
```
python3 task2_evaluate.py from-file --ref_file data/test_gold.csv pred.csv
```

### Reproduction
You have to have the full blind test data(and name it as `data/task2.csv`) to reproduce our scores.

1. Rename `data/pos.txt` to `data/pos_tags.txt`
2. Download model : [eval_bio(BERT+bio+viterbi)](https://drive.google.com/file/d/1omc-hy4uAb1JaeVrNQvbvGOQ3tZga7C3/view?usp=sharing)
3. Unzip **eval_bio.zip** file then run
```shell
bash reproduce.sh
```

### Reference
[Stanza toolkit](https://stanfordnlp.github.io/stanza/) 
example code from [huggingface](https://github.com/huggingface/transformers)

### How do I cite this work?
```
@inproceedings{Kao2020ntunlpl, 
title={NTUNLPL at FinCausal 2020, Task 2: Improving Causality Detection Using Viterbi Decoder}, 
    author = {Kao, Pei-Wei and Chen, Chung-Chi and Huang, Hen-Hsen and Chen, Hsin-Hsi}, 
    booktitle ={The 1st Joint Workshop on Financial Narrative Processing and MultiLing Financial Summarisation (FNP-FNS 2020)},
    year = {2020} 
}
```

