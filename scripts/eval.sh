# CLEVR-Change
model_dir=exp/clevr/eval
clevr_path=your_dataset_path
model_path=exp/clevr/clevr_model_params.pth

python train.py \
    --batch_size=16 \
    --consist_w=0.25 \
    --ortho_w=1.0 \
    --vit_lora_k=16 \
    --qformer_lora_k=4 \
    --max_length=50 \
    --dataset=clevr \
    --mode=test \
    --model_dir=${model_dir} \
    --clver_path=${clevr_path} \
    --model_path=${model_path}

# Spot-the-diff
model_dir=exp/spot/eval
spot_path=your_dataset_path
model_path=exp/spot/spot_model_params.pth

python train.py \
    --batch_size=16 \
    --consist_w=0.75 \
    --ortho_w=0.5 \
    --vit_lora_k=4 \
    --qformer_lora_k=4 \
    --max_length=50 \
    --dataset=spot \
    --mode=test \
    --model_dir=${model_dir} \
    --spot_path=${spot_path} \
    --model_path=${model_path}


# Image-Editing-Request
model_dir=exp/IER/eval
IER_path=your_dataset_path
model_path=exp/IER/IER_model_params.pth

python train.py \
    --batch_size=16 \
    --consist_w=0.1 \
    --ortho_w=0.75 \
    --vit_lora_k=4 \
    --qformer_lora_k=4 \
    --max_length=50 \
    --dataset=IER \
    --mode=test \
    --model_dir=${model_dir} \
    --IER_path=${IER_path} \
    --model_path=${model_path}