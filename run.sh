python3 preprocess.py data/train.csv train
python3 preprocess.py data/test.csv test
cat data/train.txt data/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > data/pos_tags.txt

export MAX_LENGTH=350
export BERT_MODEL=bert-base-cased
export DATA_DIR=data
export OUTPUT_DIR=fincausal
export BATCH_SIZE=4
export NUM_EPOCHS=1
export SAVE_STEPS=3000
export SEED=1

CUDA_VISIBLE_DEVICES=0 python3 run.py \
--data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_predict \
--add_pos \

python3 submission.py $OUTPUT_DIR/test_predictions.txt data/test.csv pred.csv