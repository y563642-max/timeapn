if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/PatchTST" ]; then
  mkdir ./logs/LongForecasting/PatchTST
fi

seq_len=96
label_len=48

gpu=1
features=M
model_name=PatchTST

# root_path_name=./datasets/weather/
# data_path_name=weather.csv
# model_id_name=weather
# data_name=custom

for pred_len in 96 192 336; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --model_id $norm_type_ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --kernel_len 1 \
    --hkernel_len 1 \
    --stride 8\
    --des 'Exp' \
    --period_len 24 \
    --station_lr 0.0004\
    --wavelet 'coif3'\
    --j 1 \
    --twice_epoch 2 \
    --pd_model 1024 \
    --pe_layers 0 \
    --pd_ff 1024\
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTh1_'$pred_len.log
  done

for pred_len in 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --model_id $norm_type_ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --n_heads 8 \
    --d_model 16 \
    --d_ff 256 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 32\
    --stride 8\
    --des 'Exp' \
    --period_len 24 \
    --station_lr 0.0001\
    --train_epochs 10\
    --patience 3\
    --wavelet 'coif3'\
    --j 1 \
    --kernel_len 1 \
    --hkernel_len 1 \
    --twice_epoch 1 \
    --pd_model 512 \
    --pe_layers 0 \
    --pd_ff 1024\
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTh1_'$pred_len.log
  done

for pred_len in 96; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm1.csv \
    --model_id $norm_type_ETTm1_96_$pred_len \
    --model $model_name \
    --data ETTm1 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --factor 3 \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --learning_rate 0.001 \
    --j 1 \
    --pe_layers 0 \
    --twice_epoch 3 \
    --batch_size 128 \
    --station_lr 0.0006 \
    --pd_ff 512 \
    --pd_model 512 \
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTm1_'$pred_len.log
  done

for pred_len in 192 ; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm1.csv \
    --model_id $norm_type_ETTm1_96_$pred_len \
    --model $model_name \
    --data ETTm1 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --factor 3 \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --learning_rate 0.001 \
    --j 1 \
    --pe_layers 0 \
    --twice_epoch 3 \
    --batch_size 128 \
    --station_lr  0.0001 \
    --pd_ff 512 \
    --pd_model 64 \
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTm1_'$pred_len.log
  done

for pred_len in 336; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm1.csv \
    --model_id $norm_type_ETTm1_96_$pred_len \
    --model $model_name \
    --data ETTm1 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --factor 3 \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --learning_rate 0.001 \
    --j 1 \
    --pe_layers 0 \
    --twice_epoch 3 \
    --batch_size 128 \
    --station_lr  0.00008 \
    --pd_ff 512 \
    --pd_model 64 \
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTm1_'$pred_len.log
  done


for pred_len in 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm1.csv \
    --model_id $norm_type_ETTm1_96_$pred_len \
    --model $model_name \
    --data ETTm1 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --factor 3 \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --learning_rate 0.001 \
    --j 1 \
    --pe_layers 0 \
    --twice_epoch 3 \
    --batch_size 128 \
    --station_lr  0.00008 \
    --pd_ff 128 \
    --pd_model 512 \
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTm1_'$pred_len.log
  done

for pred_len in 96 ; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm2.csv \
    --model_id $norm_type_ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --n_heads 16 \
    --d_model 64 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --factor 2 \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --station_lr  0.00008\
    --learning_rate 0.0006\
    --j 0 \
    --twice_epoch 4 \
    --pd_model 64 \
    --batch_size 128\
    --pe_layers 1 \
    --pd_ff 512\
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTm2_'$pred_len.log
  done

for pred_len in 192 ; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm2.csv \
    --model_id $norm_type_ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --factor 3 \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --station_lr 0.00006\
    --learning_rate 0.0001\
    --j 0 \
    --twice_epoch 3 \
    --pd_model 128 \
    --batch_size 128\
    --pe_layers 1\
    --pd_ff 1024\
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTm2_'$pred_len.log
  done
  
for pred_len in 336 ; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm2.csv \
    --model_id $norm_type_ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --factor 3 \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 1 \
    --station_lr 0.00008\
    --learning_rate 0.0008\
    --j 0 \
    --twice_epoch 3 \
    --pd_model 128 \
    --batch_size 128\
    --pe_layers 2 \
    --pd_ff 512\
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTm2_'$pred_len.log
  done

for pred_len in 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm2.csv \
    --model_id $norm_type_ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --n_heads 16 \
    --d_model 64 \
    --d_ff 256 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --factor 1 \
    --enc_in 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 1 \
    --station_lr 0.00006\
    --learning_rate 0.002\
    --j 0 \
    --twice_epoch 3 \
    --pd_model 64 \
    --batch_size 128\
    --pe_layers 2 \
    --pd_ff 512\
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTm2_'$pred_len.log
  done


for pred_len in 96 192 336; do
    CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/weather/ \
      --data_path weather.csv \
      --model_id patchtst_weather_$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --station_lr 0.0008 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --features $features \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --pd_ff 256\
      --j 1 \
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/PatchTST/$model_name'_'weather'_'$seq_len'_'$pred_len.log 
    done

for pred_len in 720; do
    CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/weather/ \
      --data_path weather.csv \
      --model_id patchtst_weather_$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --station_lr 0.0008 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --features $features \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --pd_ff 512\
      --pd_model 256\
      --j 1 \
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/PatchTST/$model_name'_'weather'_'$seq_len'_'$pred_len.log 
    done

for pred_len in 96; do
     CUDA_VISIBLE_DEVICES=$gpu \
     python -u run_longExp.py \
       --is_training 1 \
       --root_path ./dataset/ \
       --data_path electricity.csv \
       --model_id $norm_type_electricity_96_$pred_len \
       --model $model_name \
       --data custom \
       --features $features \
       --seq_len $seq_len \
       --label_len $label_len \
       --pred_len $pred_len \
       --e_layers 2 \
       --d_layers 1 \
       --factor 3 \
       --enc_in 321 \
       --dec_in 321 \
       --c_out 321 \
       --des 'Exp' \
       --period_len 24 \
       --batch_size 6 \
       --station_lr 0.0001 \
       --twice_epoch 1 \
       --pd_ff 128 \
       --pd_model 256 \
       --pe_layers 0 \
       --j 1 \
       --itr 1 >logs/LongForecasting/PatchTST/$model_name'_elec_'$pred_len.log
 done
 
for pred_len in 192; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id $norm_type_electricity_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --period_len 24 \
    --batch_size 6 \
    --station_lr 0.00008 \
    --twice_epoch 2 \
    --pd_ff  256 \
    --pd_model 1024\
    --pe_layers 0 \
    --j 1 \
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_elec_'$pred_len.log
  done

for pred_len in 336; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id $norm_type_electricity_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --period_len 24 \
    --batch_size 6 \
    --twice_epoch 1 \
    --station_lr 0.00008 \
    --pd_ff 256 \
    --pd_model 512 \
    --pe_layers 0 \
    --j 1 \
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_elec_'$pred_len.log
  done
  
for pred_len in 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id $norm_type_electricity_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --period_len 24 \
    --batch_size 6 \
    --station_lr 0.00008 \
    --pd_ff  512 \
    --twice_epoch 2 \
    --pd_model 256 \
    --pe_layers 0 \
    --j 1 \
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_elec_'$pred_len.log
  done


for pred_len in 96; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTh2.csv \
    --model_id $norm_type_ETTh2_96_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --n_heads 8 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --des 'Exp' \
    --period_len 24 \
    --twice_epoch 1 \
    --pd_ff 512 \
    --pd_model 512 \
    --station_lr 0.0001\
    --pe_layers 0 \
    --batch_size 256 \
    --j 0 \
    --kernel_len 1 \
    --learning_rate 0.0006\
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTh2_'$pred_len.log
  done

for pred_len in 192; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTh2.csv \
    --model_id $norm_type_ETTh2_96_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --n_heads 8 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --des 'Exp' \
    --period_len 24 \
    --twice_epoch 1 \
    --pd_ff 1024 \
    --pd_model 1024 \
    --station_lr 0.00008\
    --pe_layers 0 \
    --batch_size 256 \
    --learning_rate 0.0008\
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTh2_'$pred_len.log
  done
  
for pred_len in 336; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTh2.csv \
    --model_id $norm_type_ETTh2_96_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --des 'Exp' \
    --period_len 24 \
    --twice_epoch 2 \
    --pd_ff 1024 \
    --pd_model 512 \
    --station_lr 0.00006\
    --pe_layers 1 \
    --batch_size 128 \
    --learning_rate 0.0001\
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTh2_'$pred_len.log
  done

for pred_len in 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTh2.csv \
    --model_id $norm_type_ETTh2_96_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --stride 14\
    --des 'Exp' \
    --period_len 24 \
    --twice_epoch 3 \
    --pd_ff 512 \
    --pd_model 256 \
    --station_lr 0.00004\
    --wavelet 'coif6'\
    --pe_layers 2 \
    --batch_size 128 \
    --learning_rate 0.001\
    --itr 1 >logs/LongForecasting/PatchTST/$model_name'_ETTh2_'$pred_len.log
  done


for pred_len in 96 ; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/traffic \
    --data_path traffic.csv \
    --model_id $norm_type_traffic_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 720 \
    --label_len 168 \
    --pred_len $pred_len \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --kernel_len 1 \
    --hkernel_len 1 \
    --learning_rate 0.001 \
    --station_lr 0.0001\
    --twice_epoch 2 \
    --batch_size 2\
    --j 0 \
    --pd_model 1024 \
    --pd_ff 1024 >logs/LongForecasting/PatchTST/$model_name'_traf_'$pred_len.log
  done

for pred_len in 192 ; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/traffic \
    --data_path traffic.csv \
    --model_id $norm_type_traffic_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 720 \
    --label_len 168 \
    --pred_len $pred_len \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --kernel_len 1 \
    --hkernel_len 1 \
    --learning_rate 0.0006 \
    --station_lr 0.0001\
    --twice_epoch 4 \
    --batch_size 2\
    --j 1 \
    --pd_model 1024 \
    --pd_ff 1024 >logs/LongForecasting/PatchTST/$model_name'_traf_'$pred_len.log
  done
  

for pred_len in 336 720 ; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/traffic \
    --data_path traffic.csv \
    --model_id $norm_type_traffic_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 720 \
    --label_len 168 \
    --pred_len $pred_len \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --kernel_len 1 \
    --hkernel_len 1 \
    --learning_rate 0.001 \
    --station_lr 0.0001\
    --twice_epoch 4 \
    --batch_size 2\
    --j 1 \
    --pd_model 1024 \
    --pd_ff 1024 >logs/LongForecasting/PatchTST/$model_name'_traf_'$pred_len.log
  done