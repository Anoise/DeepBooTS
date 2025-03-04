export CUDA_VISIBLE_DEVICES=1

root_path=data/TS/ETT-small/
model_name=DeepBooTS
dataset=ETTh1
seq_len=96

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 2 \
  --attn 0 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'96.log  

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 3 \
  --attn 0 \
  --learning_rate 0.00005 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'192.log 

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers 6 \
 --attn 0 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'336.log 

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers 6 \
  --attn 0 \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'720.log 