if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/DLinear" ]; then
  mkdir ./logs/LongForecasting/DLinear
fi

gpu=0
features=M
model_name=DLinear

for pred_len in 96 192; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1_336_$pred_len$model_name \
    --model $model_name \
    --data ETTh1 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 24 \
    --station_lr 0.0004 \
    --j 1 \
    --pe_layers 0 \
    --pd_model 256 \
    --itr 1 \
    --pd_ff 512 >logs/LongForecasting/DLinear/$model_name'_eh1_'$pred_len.log
    done

for pred_len in 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1_336_$pred_len$model_name \
    --model $model_name \
    --data ETTh1 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 24 \
    --station_lr 0.0004 \
    --j 0 \
    --pe_layers 0 \
    --pd_model 256 \
    --itr 1 \
    --pd_ff 512 >logs/LongForecasting/DLinear/$model_name'_eh1_'$pred_len.log
    done

for pred_len in 96 192 336; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_336_$pred_len$model_name \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 24 \
    --station_lr 0.00004 \
    --learning_rate 0.00008\
    --j 1 \
    --pe_layers 0 \
    --pd_model 1024 \
    --itr 1 \
    --pd_ff 128 >logs/LongForecasting/DLinear/$model_name'_eh1_'$pred_len.log
    done

for pred_len in 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_336_$pred_len$model_name \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 24 \
    --station_lr 0.00002 \
    --j 1 \
    --pe_layers 0 \
    --pd_model 512 \
    --itr 1 \
    --pd_ff 128 >logs/LongForecasting/DLinear/$model_name'_eh1_'$pred_len.log
    done

 for pred_len in 96; do
   CUDA_VISIBLE_DEVICES=$gpu \
   python -u run_longExp.py \
     --is_training 1 \
     --root_path ./datasets/ETT/ \
     --data_path ETTm1.csv \
     --model_id ETTm1_336_$pred_len$model_name \
     --model $model_name \
     --data ETTm1 \
     --features $features \
     --seq_len 336 \
     --label_len 168 \
     --pred_len $pred_len \
     --enc_in 7 \
     --des 'Exp' \
     --period_len 12 \
     --kernel_len 12 \
     --twice_epoch 3\
     --wavelet 'coif6'\
     --station_lr 0.00006\
     --learning_rate 0.00006 \
     --pd_model 256\
     --pd_ff 512 \
     --itr 1 >logs/LongForecasting/DLinear/$model_name'_em1_'$pred_len.log
   done

 for pred_len in 192; do
   CUDA_VISIBLE_DEVICES=$gpu \
   python -u run_longExp.py \
     --is_training 1 \
     --root_path ./datasets/ETT/ \
     --data_path ETTm1.csv \
     --model_id ETTm1_336_$pred_len$model_name \
     --model $model_name \
     --data ETTm1 \
     --features $features \
     --seq_len 336 \
     --label_len 168 \
     --pred_len $pred_len \
     --enc_in 7 \
     --des 'Exp' \
     --twice_epoch 3\
     --wavelet 'coif6'\
     --period_len 12 \
     --kernel_len 12 \
     --station_lr 0.00006\
     --learning_rate 0.00006 \
     --twice_epoch 2\
     --pd_model 128\
     --pd_ff 1024 \
     --itr 1 >logs/LongForecasting/DLinear/$model_name'_em1_'$pred_len.log
   done

 for pred_len in 336; do
   CUDA_VISIBLE_DEVICES=$gpu \
   python -u run_longExp.py \
     --is_training 1 \
     --root_path ./datasets/ETT/ \
     --data_path ETTm1.csv \
     --model_id ETTm1_336_$pred_len$model_name \
     --model $model_name \
     --data ETTm1 \
     --features $features \
     --seq_len 336 \
     --label_len 168 \
     --pred_len $pred_len \
     --enc_in 7 \
     --des 'Exp' \
     --twice_epoch 3\
     --wavelet 'coif6'\
     --period_len 12 \
     --kernel_len 12 \
     --station_lr 0.00006\
     --learning_rate 0.00006 \
     --twice_epoch 2\
     --pd_model 128\
     --pd_ff 512 \
     --itr 1 >logs/LongForecasting/DLinear/$model_name'_em1_'$pred_len.log
   done

 for pred_len in 720; do
   CUDA_VISIBLE_DEVICES=$gpu \
   python -u run_longExp.py \
     --is_training 1 \
     --root_path ./datasets/ETT/ \
     --data_path ETTm1.csv \
     --model_id ETTm1_336_$pred_len$model_name \
     --model $model_name \
     --data ETTm1 \
     --features $features \
     --seq_len 336 \
     --label_len 168 \
     --pred_len $pred_len \
     --enc_in 7 \
     --des 'Exp' \
     --twice_epoch 3\
     --wavelet 'coif6'\
     --period_len 12 \
     --kernel_len 12 \
     --station_lr 0.00006\
     --learning_rate 0.00006 \
     --twice_epoch 2\
     --pd_model 128\
     --pd_ff 1024 \
     --itr 1 >logs/LongForecasting/DLinear/$model_name'_em1_'$pred_len.log
   done

for pred_len in 96; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_336_$pred_len$model_name \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --batch_size 128\
    --kernel_len 6 \
    --station_lr 0.00008\
    --pd_ff 256 \
    --pd_model 256 \
    --itr 1 >logs/LongForecasting/DLinear/$model_name'_em2_'$pred_len.log
  done

for pred_len in 192; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_336_$pred_len$model_name \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --batch_size 128\
    --kernel_len 6 \
    --station_lr 0.00008\
    --learning_rate 0.0006\
    --pd_ff 512 \
    --pd_model 256 \
    --itr 1 >logs/LongForecasting/DLinear/$model_name'_em2_'$pred_len.log
  done

for pred_len in 336; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_336_$pred_len$model_name \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --batch_size 128\
    --kernel_len 6 \
    --station_lr 0.00008\
    --pd_ff 128 \
    --pd_model 512 \
    --itr 1 >logs/LongForecasting/DLinear/$model_name'_em2_'$pred_len.log
  done

for pred_len in 720; do
 CUDA_VISIBLE_DEVICES=$gpu \
 python -u run_longExp.py \
   --is_training 1 \
   --root_path ./datasets/ETT/ \
   --data_path ETTm2.csv \
   --model_id ETTm2_336_$pred_len$model_name \
   --model $model_name \
   --data ETTm2 \
   --features $features \
   --seq_len 336 \
   --label_len 168 \
   --pred_len $pred_len \
   --enc_in 7 \
   --des 'Exp' \
   --period_len 12 \
   --batch_size 128\
   --kernel_len 6 \
   --station_lr 0.00008\
   --pd_ff 128 \
   --pd_model 1024 \
   --itr 1 >logs/LongForecasting/DLinear/$model_name'_em2_'$pred_len.log
 done

for pred_len in 96 192 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/weather \
    --data_path weather.csv \
    --model_id weather_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --itr 1 \
    --period_len 12 \
    --pe_layers 2 \
    --station_lr 0.0008 \
    --j 1\
    --pd_ff 128 \
    --pd_model 128 >logs/LongForecasting/DLinear/$model_name'_wea_'$pred_len.log
  done

for pred_len in 336; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/weather \
    --data_path weather.csv \
    --model_id weather_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --itr 1 \
    --period_len 12 \
    --pe_layers 2 \
    --station_lr 0.0006 \
    --j 1 \
    --pd_ff 128 \
    --pd_model 128 >logs/LongForecasting/DLinear/$model_name'_wea_'$pred_len.log
  done


for pred_len in 96; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --station_lr 0.001 \
    --period_len 24 \
    --j 1 \
    --pd_ff 512 \
    --twice_epoch 2 \
    --pd_model 1024 \
    --itr 1 >logs/LongForecasting/DLinear/$model_name'_elc_'$pred_len.log
  done

for pred_len in 192; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --station_lr 0.0008 \
    --period_len 24 \
    --wavelet 'coif6'\
    --j 1 \
    --pd_ff 1024 \
    --pd_model 1024 \
    --itr 1 >logs/LongForecasting/DLinear/$model_name'_elc_'$pred_len.log
  done

for pred_len in 336; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --station_lr  0.0007 \
    --period_len 24 \
    --wavelet 'coif6'\
    --j 1 \
    --pd_ff 1024 \
    --pd_model 1024 \
    --itr 1 >logs/LongForecasting/DLinear/$model_name'_elc_'$pred_len.log
  done

for pred_len in 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --station_lr 0.0004 \
    --period_len 24 \
    --j 1 \
    --pd_ff 256 \
    --pd_model 512 \
    --itr 1 >logs/LongForecasting/DLinear/$model_name'_elc_'$pred_len.log
  done

for pred_len in 96 192 336 ; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/traffic \
    --data_path traffic.csv \
    --model_id traffic_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --period_len 24 \
    --learning_rate 0.001 \
    --station_lr 0.0008\
    --twice_epoch 2 \
    --j 1 \
    --kernel_len 1 \
    --hkernel_len 1 \
    --pd_model 1024 \
    --batch_size 64 \
    --pd_ff 1024 \
    --wavelet 'coif6' >logs/LongForecasting/DLinear/$model_name'_tra_'$pred_len.log
  done

for pred_len in 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/traffic \
    --data_path traffic.csv \
    --model_id traffic_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --period_len 24 \
    --kernel_len 1 \
    --hkernel_len 1 \
    --learning_rate 0.0006 \
    --station_lr 0.0008\
    --twice_epoch 4 \
    --j 1 \
    --pd_model 1024 \
    --pd_ff 1024 \
    --wavelet 'coif3' >logs/LongForecasting/DLinear/$model_name'_tra_'$pred_len.log
  done







