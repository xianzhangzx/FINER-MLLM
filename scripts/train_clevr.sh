model_dir=exp/clevr/
clevr_path=your_dataset_path

python train.py \
    --batch_size=16 \
    --num_epochs=40 \
    --lr=1e-5 \
    --min_lr=1e-5 \
    --peak_lr=5e-5 \
    --warmup_steps=4000 \
    --consist_w=0.25 \
    --ortho_w=1.0 \
    --vit_lora_k=16 \
    --qformer_lora_k=4 \
    --weight_decay=0.1 \
    --max_length=50 \
    --dataset=clevr \
    --mode=train \
    --model_dir=${model_dir} \
    --clver_path=${clevr_path}