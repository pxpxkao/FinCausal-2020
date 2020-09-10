python3 preprocess.py data/task2.csv test

export MAX_LENGTH=350
export BERT_MODEL=eval_bio

export DATA_DIR=data
export OUTPUT_DIR=pred
export SEED=1

CUDA_VISIBLE_DEVICES=0 python3 run.py \
--data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--seed $SEED \
--do_predict \

python3 submission.py $OUTPUT_DIR/test_predictions.txt data/task2.csv reproduce.csv