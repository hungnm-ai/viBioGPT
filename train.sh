python -m training \
    --model_name_or_path Viet-Mistral/Vistral-7B-Chat \
    --train_path ./data/qa_pairs_edoctor.json,./data/qa_vinmec.json \
    --lora True \
    --qlora True \
    --bf16 True \
    --output_dir models/bioGPT-instruct \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 3 \
    --eval_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --eval_steps 40 \
    --save_strategy "epoch" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1.2e-5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --packing False \
    --report_to "wandb"