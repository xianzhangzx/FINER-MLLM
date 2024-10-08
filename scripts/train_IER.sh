model_dir=exp/IER/
IER_path=your_dataset_path

python train.py \
    --batch_size=16 \
    --num_epochs=20 \
    --lr=1e-5 \
    --min_lr=1e-5 \
    --peak_lr=3e-5 \
    --warmup_steps=250 \
    --consist_w=0.1 \
    --ortho_w=0.75 \
    --vit_lora_k=4 \
    --qformer_lora_k=4 \
    --weight_decay=0.1 \
    --max_length=50 \
    --dataset=IER \
    --mode=train \
    --model_dir=${model_dir} \
    --IER_path=${IER_path}