export CUDA_VISIBLE_DEVICES=0

model_name=DeepBooTS
root_path=data/TS/electricity
seq_len=336

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_336_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 6 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 0.0001\
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_'96.log  

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_336_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001\
  --attn 0 \
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_'192.log  


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_336_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_'336.log 


python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path electricity.csv \
  --model_id ECL_336_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --itr 1 >logs/$model_name'_'electricity_$seq_len'_'720.log 