mkdir -p ../trainer_output/ss3/log

nohup sh ../run_main.sh \
      --model "../trainer_output/checkpoint-3/encoder" \
      --tokenizer_name "Rostlab/prot_bert" \
      --output_file ss3-GLProtein \
      --task_name ss3 \
      --do_train True \
      --epoch 5 \
      --optimizer AdamW \
      --per_device_batch_size 2 \
      --gradient_accumulation_steps 16 \
      --eval_step 50 \
      --eval_batchsize 4 \
      --warmup_ratio 0.08 \
      --learning_rate 3e-5 \
      --seed 3 \
      --frozen_bert False > ../trainer_output/ss3/log/GLProtein.out 2>&1