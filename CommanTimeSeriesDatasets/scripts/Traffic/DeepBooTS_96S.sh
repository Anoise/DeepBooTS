export CUDA_VISIBLE_DEVICES=0

root_path=data/TS/traffic
model_name=DeepBooTS
seq_len=96

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --learning_rate 0.001\
  --itr 1 >logs/$model_name'_'traffic_$seq_len'_S_'96.log  


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --learning_rate 0.001\
  --itr 1  >logs/$model_name'_'traffic_$seq_len'_S_'192.log  

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001\
  --itr 1 >logs/$model_name'_'traffic_$seq_len'_S_'336.log  

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers 6 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --batch_size 16\
  --learning_rate 0.001\
  --itr 1 >logs/$model_name'_'traffic_$seq_len'_S_'720.log 