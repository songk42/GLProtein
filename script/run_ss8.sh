mkdir -p ../outputs/ss8/log

nohup bash ../run_main.sh \
      --model "../outputs/glprotein_full/checkpoint-100000/encoder" \
      --tokenizer_name "../outputs/glprotein_full/checkpoint-100000/protein_tokenizer" \
      --output_file ss8-GLProtein \
      --task_name ss8 \
      --do_train True \
      --epoch 5 \
      --optimizer AdamW \
      --per_device_batch_size 1 \
      --gradient_accumulation_steps 32 \
      --eval_step 50 \
      --eval_batchsize 4 \
      --warmup_ratio 0.08 \
      --learning_rate 3e-5 \
      --seed 3 \
      --frozen_bert False > ../outputs/ss8/log/GLProtein.out 2>&1