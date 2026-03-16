if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/FEDformer" ]; then
  mkdir ./logs/LongForecasting/FEDformer
fi

seq_len=96
label_len=48
features=M
gpu=0
model_name=FEDformer

for pred_len in 96; do
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
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --pre_epoch 6 \
    --station_lr 0.0004\
    --j 1 \
    --twice_epoch 2 \
    --wavelet 'coif3'\
    --pe_layers 0 \
    --pd_model 512 \
    --pd_ff 1024\
    --batch_size 256\
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTh1_'$pred_len.log
  done

for pred_len in 192; do
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
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --pre_epoch 6 \
    --station_lr 0.0004\
    --j 1 \
    --twice_epoch 1 \
    --pe_layers 0 \
    --pd_model 256 \
    --pd_ff 256\
    --batch_size 128\
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTh1_'$pred_len.log
  done

for pred_len in 336; do
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
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --pre_epoch 6 \
    --station_lr 0.0004\
    --j 1 \
    --twice_epoch 1 \
    --pe_layers 0 \
    --pd_model 256 \
    --pd_ff 256\
    --batch_size 128\
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTh1_'$pred_len.log
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
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --pre_epoch 6 \
    --station_lr 0.0004\
    --j 1 \
    --twice_epoch 2 \
    --wavelet 'coif3'\
    --pe_layers 0 \
    --pd_model 512 \
    --pd_ff 1024\
    --batch_size 64\
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTh1_'$pred_len.log
  done

for pred_len in 96; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT  \
    --data_path ETTh2.csv \
    --model_id $norm_type_ETTh2_96_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --kernel_len 12 \
    --twice_epoch 0 \
    --station_lr 0.0004\
    --pd_ff 1024 \
    --pd_model 512 \
    --pe_layers 0 \
    --batch_size 256 \
    --j 1\
    --learning_rate 0.00008\
    --dr 0.3 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTh2_'$pred_len.log
  done
   
for pred_len in 192; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/  \
    --data_path ETTh2.csv \
    --model_id $norm_type_ETTh2_96_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --kernel_len 1 \
    --twice_epoch 0 \
    --station_lr 0.0006\
    --pd_ff 64 \
    --pd_model 1024 \
    --pe_layers 0 \
    --batch_size 256 \
    --learning_rate 0.00008\
    --dr 0.3 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTh2_'$pred_len.log
  done

for pred_len in 336; do
           CUDA_VISIBLE_DEVICES=$gpu \
           python -u run_longExp.py \
             --is_training 1 \
             --root_path ./datasets/ETT/  \
             --data_path ETTh2.csv \
             --model_id $norm_type_ETTh2_96_$pred_len \
             --model $model_name \
             --data ETTh2 \
             --features $features \
             --seq_len $seq_len \
             --label_len $label_len \
             --pred_len $pred_len \
             --dr 0.3 \
             --e_layers 2 \
             --d_layers 1 \
             --factor 1 \
             --enc_in 7 \
             --dec_in 7 \
             --c_out 7 \
             --des 'Exp' \
             --period_len 24 \
             --kernel_len 12 \
             --station_lr 0.0001\
             --pd_ff 1024 \
             --pd_model 512 \
             --pe_layers 1 \
             --batch_size  256 \
             --j 1\
             --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTh2_'$pred_len.log
 done
   
   
for pred_len in 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run_longExp.py \
          --is_training 1 \
          --root_path ./datasets/ETT/  \
          --data_path ETTh2.csv \
          --model_id $norm_type_ETTh2_96_$pred_len \
          --model $model_name \
          --data ETTh2 \
          --features $features \
          --seq_len $seq_len \
          --label_len $label_len \
          --pred_len $pred_len \
          --dr 0.3\
          --e_layers 2 \
          --d_layers 1 \
          --factor 1 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --period_len 24 \
          --kernel_len 12 \
          --station_lr 0.0001\
          --pd_ff 1024 \
          --pd_model 512 \
          --pe_layers 1 \
          --batch_size 128 \
          --twice_epoch 2\
          --j 0\
          --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTh2_'$pred_len.log
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
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --hkernel_len 7 \
    --station_lr 0.001\
    --dr 0.2 \
    --pd_ff 128 \
    --pd_model 512 \
    --batch_size 256\
    --twice_epoch 0 \
    --j 1 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTm1_'$pred_len.log
  done

for pred_len in 192; do
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
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --hkernel_len 7 \
    --station_lr 0.0008\
    --dr 0.2 \
    --pd_ff 128 \
    --pd_model 128 \
    --batch_size 256\
    --twice_epoch 0 \
    --j 1 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTm1_'$pred_len.log
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
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --hkernel_len 7 \
    --station_lr  0.0004\
    --dr 0.2 \
    --pd_ff 128 \
    --pd_model 256 \
    --batch_size 256\
    --twice_epoch 0 \
    --j 1 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTm1_'$pred_len.log
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
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --hkernel_len 7 \
    --station_lr 0.0006\
    --dr 0.2 \
    --pd_ff 512 \
    --pd_model 64 \
    --batch_size 128\
    --twice_epoch 0 \
    --j 1 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTm1_'$pred_len.log
  done

for pred_len in 96; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm2.csv \
    --model_id $norm_type_ETTm2_96_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --pe_layers 1 \
    --kernel_len 1 \
    --twice_epoch 1 \
    --j 0\
    --batch_size 256\
    --station_lr 0.0004\
    --pd_model 256 \
    --pd_ff 1024 \
    --dr 0.3 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTm2_'$pred_len.log
  done

for pred_len in  192; do
 CUDA_VISIBLE_DEVICES=$gpu \
 python -u run_longExp.py \
   --is_training 1 \
   --root_path ./datasets/ETT/ \
   --data_path ETTm2.csv \
   --model_id $norm_type_ETTm2_96_$pred_len \
   --model $model_name \
   --data ETTm2 \
   --features $features \
   --seq_len $seq_len \
   --label_len $label_len \
   --pred_len $pred_len \
   --e_layers 2 \
   --d_layers 1 \
   --factor 3 \
   --enc_in 7 \
   --dec_in 7 \
   --c_out 7 \
   --des 'Exp' \
   --period_len 12 \
   --pe_layers 1 \
   --j 1\
   --batch_size 256\
   --station_lr 0.0001\
   --pd_model 512 \
   --pd_ff 256 \
   --dr 0.3 \
   --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTm2_'$pred_len.log
 done

for pred_len in  336; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm2.csv \
    --model_id $norm_type_ETTm2_96_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --pe_layers 1 \
    --kernel_len 1 \
    --j 0\
    --batch_size 256\
    --station_lr 0.0008\
    --pd_model 512 \
    --pd_ff 128 \
    --dr 0.3 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTm2_'$pred_len.log
  done

for pred_len in  720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT/ \
    --data_path ETTm2.csv \
    --model_id $norm_type_ETTm2_96_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --pe_layers 1 \
    --batch_size 128\
    --station_lr  0.00008\
    --pd_model 512 \
    --pd_ff 64 \
    --j 1 \
    --dr 0.3 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_ETTm2_'$pred_len.log
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
       --e_layers 3 \
       --d_layers 1 \
       --factor 4 \
       --enc_in 321 \
       --dec_in 321 \
       --c_out 321 \
       --des 'Exp' \
       --period_len 24 \
       --pd_model 256 \
       --station_lr 0.0008\
       --pd_ff 128 \
       --twice_epoch 1 \
       --j 1 \
       --pe_layers 1 \
       --itr 1 >logs/LongForecasting/FEDformer/$model_name'_elec_'$pred_len.log
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
     --factor 4 \
     --enc_in 321 \
     --dec_in 321 \
     --c_out 321 \
     --des 'Exp' \
     --period_len 24 \
     --twice_epoch 1 \
     --pd_model 512 \
     --station_lr 0.001\
     --pd_ff 64 \
     --j 1 \
     --pe_layers 0 \
     --itr 1 >logs/LongForecasting/FEDformer/$model_name'_elec_'$pred_len.log
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
     --factor 4 \
     --enc_in 321 \
     --dec_in 321 \
     --c_out 321 \
     --des 'Exp' \
     --period_len 24 \
     --pd_model 1024 \
     --station_lr 0.0001\
     --pd_ff 64 \
     --pe_layers 0 \
     --itr 1 >logs/LongForecasting/FEDformer/$model_name'_elec_'$pred_len.log
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
     --pd_model 1024 \
     --station_lr 0.001\
     --pd_ff 128 \
     --pe_layers 0 \
     --itr 1 >logs/LongForecasting/FEDformer/$model_name'_elec_'$pred_len.log
   done


for pred_len in 96; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/weather \
    --data_path weather.csv \
    --model_id $norm_type_weather_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --period_len 12 \
    --j 1 \
    --station_lr 0.0008 \
    --pd_ff 1024\
    --pd_model 512\
    --batch_size 128\
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_wea_'$pred_len.log
  done

for pred_len in 192; do
 CUDA_VISIBLE_DEVICES=$gpu \
 python -u run_longExp.py \
   --is_training 1 \
   --root_path ./datasets/weather \
   --data_path weather.csv \
   --model_id $norm_type_weather_96_$pred_len \
   --model $model_name \
   --data custom \
   --features $features \
   --seq_len $seq_len \
   --label_len $label_len \
   --pred_len $pred_len \
   --e_layers 2 \
   --d_layers 1 \
   --factor 3 \
   --enc_in 21 \
   --dec_in 21 \
   --c_out 21 \
   --des 'Exp' \
   --period_len 12 \
   --j 1 \
   --station_lr 0.0001 \
   --pd_ff 1024\
   --pd_model 512\
   --kernel_len 1 \
   --batch_size 128\
   --itr 1 >logs/LongForecasting/FEDformer/$model_name'_wea_'$pred_len.log
 done

for pred_len in 336; do
 CUDA_VISIBLE_DEVICES=$gpu \
 python -u run_longExp.py \
   --is_training 1 \
   --root_path ./datasets/weather \
   --data_path weather.csv \
   --model_id $norm_type_weather_96_$pred_len \
   --model $model_name \
   --data custom \
   --features $features \
   --seq_len $seq_len \
   --label_len $label_len \
   --pred_len $pred_len \
   --e_layers 2 \
   --d_layers 1 \
   --factor 3 \
   --enc_in 21 \
   --dec_in 21 \
   --c_out 21 \
   --des 'Exp' \
   --period_len 12 \
   --j 0 \
   --station_lr 0.001 \
   --train_epochs 5\
   --twice_epoch 1 \
   --pd_ff 256\
   --pd_model 256\
   --batch_size 128\
   --itr 1 >logs/LongForecasting/FEDformer/$model_name'_wea_'$pred_len.log
 done


 for pred_len in 720; do
   CUDA_VISIBLE_DEVICES=$gpu \
   python -u run_longExp.py \
     --is_training 1 \
     --root_path ./datasets/weather \
     --data_path weather.csv \
     --model_id $norm_type_weather_96_$pred_len \
     --model $model_name \
     --data custom \
     --features $features \
     --seq_len $seq_len \
     --label_len 48 \
     --pred_len $pred_len \
     --e_layers 2 \
     --d_layers 1 \
     --factor 3 \
     --enc_in 21 \
     --dec_in 21 \
     --c_out 21 \
     --des 'Exp' \
     --period_len 12 \
     --j 1 \
     --station_lr 0.0006 \
     --train_epochs 5\
     --learning_rate 0.0001\
     --twice_epoch 1 \
     --pd_ff 1024 \
     --pd_model 1024\
     --itr 1 >logs/LongForecasting/FEDformer/$model_name'_wea_'$pred_len.log
   done

for pred_len in 96 ; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path  ./datasets/traffic \
    --data_path traffic.csv \
    --model_id trafffic_96_96$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --period_len 24 \
    --twice_epoch 2 \
    --kernel_len 1 \
    --hkernel_len 1 \
    --station_lr 0.0002\
    --pd_ff 1024 \
    --pd_model 1024 \
    --pe_layers 2 \
    --j 0 \
    --learning_rate 0.0004\
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_traf_'$pred_len.log
  done

for pred_len in 192; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id trafffic_96_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --label_len 168 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --period_len 24 \
    --twice_epoch 3 \
    --station_lr 0.0004\
    --kernel_len 1 \
    --hkernel_len 1 \
    --pd_ff 128 \
    --pd_model 256\
    --pe_layers 1\
    --learning_rate 0.0008\
    --batch_size 64\
    --j 1 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_traf_'$pred_len.log
  done

  for pred_len in 336; do
    CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/traffic/ \
      --data_path traffic.csv \
      --model_id trafffic_96_$pred_len$model_name \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 96 \
      --label_len 168 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --period_len 24 \
      --twice_epoch 3 \
      --station_lr 0.0006\
      --kernel_len 1 \
      --hkernel_len 1 \
      --pd_ff 1024 \
      --pd_model 512\
      --pe_layers 1\
      --batch_size 64\
      --j 1 \
      --itr 1 >logs/LongForecasting/FEDformer/$model_name'_traf_'$pred_len.log
    done

for pred_len in 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id trafffic_96_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 96 \
    --label_len 168 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --period_len 24 \
    --twice_epoch 3 \
    --kernel_len 1 \
    --hkernel_len 1 \
    --station_lr 0.0004\
    --pd_ff 256 \
    --pd_model 128\
    --batch_size 64\
    --learning_rate 0.0008\
    --pe_layers 0\
    --j 1 \
    --itr 1 >logs/LongForecasting/FEDformer/$model_name'_traf_'$pred_len.log
 done 

