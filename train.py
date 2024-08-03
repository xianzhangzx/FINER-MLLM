import os 
import sys
import argparse
import logging
import warnings 
import json 
import random

import numpy as np 
import torch 
from torch.utils.data import dataloader
from tqdm import tqdm

import utils
import datasets
import model as diff_model
from lavis.models import load_preprocess 
from omegaconf import OmegaConf
from torch.cuda.amp import autocast as autocast, GradScaler
from lavis.common.optims import LinearWarmupCosineLRScheduler

os.environ['TMPDIR'] = '/gpushare/IDC/tmp'
warnings.filterwarnings("ignore")
torch.set_num_threads(8)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)  
parser.add_argument('--dataset', default = 'clevr', help = "data set type")
parser.add_argument('--spot_path', default = 'your dataset dir')
parser.add_argument('--clver_path', default='your dataset dir')
parser.add_argument('--IER_path', default = 'your dataset dir')
parser.add_argument('--spot_rag_path', default='rag_store/spot_filter_change_retrieved_caps.json') # rag_store/spot_change_retrieved_caps.json
parser.add_argument('--clver_rag_path', default="rag_store/clevr_filter_change_retrieved_caps.json") # rag_store/clevr_change_retrieved_caps.json
parser.add_argument('--IER_rag_path', default = 'rag_store/IER_change_retrieved_caps.json')
parser.add_argument('--model_pth', default = '/zhang_xian/model_pth/Finer-MLLM/clevr_model_params.pth')
# parser.add_argument('--model_pth', default = '/zhang_xian/model_pth/Finer-MLLM/spot_model_params.pth')
parser.add_argument('--mode', default = 'train') 

parser.add_argument('--consist_w', type=float, default=0.25)
parser.add_argument('--ortho_w', type=float, default=1.0)
parser.add_argument('--vit_lora_k', type=int, default=16)
parser.add_argument('--qformer_lora_k', type=int, default=4)
parser.add_argument('--warmup_steps', type=int, default=4000)

parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--min_lr', type=float, default=1e-5)
parser.add_argument('--peak_lr', type=float, default=4e-5)

parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--eval_frequency', type=int, default=1)
parser.add_argument('--early_stop_num', type=int, default=10)
parser.add_argument('--prompt', type=str, 
                    default="the difference between the before image and after image is that")

parser.add_argument('--model_type', type=str, default='vicuna7b') # caption_coco_opt2.7b
parser.add_argument('--model_dir', default='exp/clevr_k10', help="save results")
parser.add_argument('--gt_dir', default='./eval_data', help='the ground-truth caption')
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()

def get_dataset(data_name, split):
    cfg = OmegaConf.load('configs/blip2_instruct_vicuna7b.yaml')
    img_preprocess, _ = load_preprocess(cfg.preprocess)
    transform = img_preprocess['eval']
    
    if data_name == 'clevr':
        dataset = datasets.CLEVR_Dataset(data_path=args.clver_path, 
                                         transform=transform, 
                                         split=split, 
                                         prompt=args.prompt,
                                         rag_store_path=args.clver_rag_path,
                                         )
        return dataset
    elif data_name == 'spot':
        dataset = datasets.SpotDataset(image_path=os.path.join(args.spot_path, 'resized_images'),
                                            anno_path=os.path.join(args.spot_path, 'captions'),
                                            racap_path=args.spot_rag_path,
                                            transform=transform,
                                            split=split,
                                            prompt=args.prompt)
        return dataset
    elif data_name == "IER":
        dataset = datasets.Image_Edit_Request(data_path=args.IER_path,
                                              transform=transform,
                                              split=split,
                                              prompt=args.prompt,
                                              rag_store_path=args.IER_rag_path)
        return dataset


def create_model_and_optimizer(need_optim=True):
    model = diff_model.FINER_MLLM.load_pretrained_model_from_blip2(model_type=args.model_type, consist_w=args.consist_w, ortho_w=args.ortho_w, vit_lora_k=args.vit_lora_k, qformer_lora_k=args.qformer_lora_k)
    model.cuda()
    if need_optim:
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        return model, optimizer
    else:
        return model

def train(model, optimizer, dataloader, scaler, cur_epoch, cur_step, scheduler):
    model.train()
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader), mininterval=60, disable=False) as t:

        for step, data in enumerate(dataloader):
            # step update lr
            if cur_step < args.warmup_steps:
                scheduler.step(0, cur_step)
            else:
                skip_epoch = int(args.warmup_steps // len(dataloader))
                scheduler.step(cur_epoch - skip_epoch, cur_step)
            cur_step += 1
            
            if args.dataset == "spot":
                bef_imgs = data['bef_img'].cuda()
                aft_imgs = data['aft_img'].cuda()
                captions = data['caption']
                ref_tokens = data['lexicon']

                optimizer.zero_grad()
                with autocast():
                    loss = model(bef_imgs, aft_imgs, captions, ref_tokens)
            
            elif args.dataset == "clevr":
                bef_imgs = data['bef_image'].cuda()
                aft_imgs = data['aft_image'].cuda()
                nsc_imgs = data['nsc_image'].cuda()
                sc_captions = data['sc_caption']
                nsc_captions = data['nsc_caption']
                sc_ref_tokens = data['sc_ref_words']
                nsc_ref_tokens = data['nsc_ref_words']
                optimizer.zero_grad()
                with autocast():
                    sc_loss = model(bef_imgs, aft_imgs, sc_captions, ref_tokens=sc_ref_tokens,)
                    nsc_loss = model(bef_imgs, nsc_imgs, nsc_captions, ref_tokens=nsc_ref_tokens,)
                    loss = sc_loss + nsc_loss
            
            elif args.dataset == "IER":
                bef_imgs = data['bef_img'].cuda()
                aft_imgs = data['aft_img'].cuda()
                captions = data['caption']
                ref_tokens = data['lexicon']
                optimizer.zero_grad()
                with autocast():
                    loss = model(bef_imgs, aft_imgs, captions, ref_tokens)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
             
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg(), cur_step

def train_and_evaluate(model, optimizer, trainset, valset):
    trainloader = dataloader.DataLoader(trainset, 
                                        shuffle=True, 
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers)

    current_best_score = float('-inf')
    scaler = GradScaler()
    epoches = args.num_epochs
    early_stop = 0
    best_epoch = -1
    best_epoch_score = None
    best_model_path = ""
    cur_step = 0

    scheduler = LinearWarmupCosineLRScheduler(optimizer, max_epoch=args.num_epochs, min_lr=args.min_lr, init_lr=args.peak_lr, warmup_steps=args.warmup_steps, warmup_start_lr=args.lr)
    
    for epoch in range(epoches):  
        early_stop += 1
        logging.info("Epoch {}/{}".format(epoch + 1, epoches))

        the_loss, cur_step = train(model, optimizer, trainloader, scaler, epoch, cur_step, scheduler)
        logging.info("loss={:05.3f}".format(the_loss)) 

        if (epoch+1) % args.eval_frequency == 0:
            score = eval_on_single_gpu(model, valset)

        logging.info("Epoch %s" % (epoch + 1))
        logging.info(score)

        if current_best_score < float(score['CIDEr']):
            current_best_score = float(score['CIDEr'])
            early_stop = 0
            best_model_path = save_model(model, epoch+1)
            best_epoch = epoch + 1
            best_epoch_score = score

        if early_stop == args.early_stop_num:
            logging.info("early stop at epoch {}.".format(epoch + 1))
            logging.info("Best Epoch is %s" % best_epoch)
            logging.info(best_epoch_score)
            logging.info("model checkpoint saved at %s" % best_model_path)
            break

    if early_stop != args.early_stop_num:
        logging.info("Best Epoch is %s" % best_epoch)
        logging.info(best_epoch_score)
        logging.info("model checkpoint saved at %s" % best_model_path)


def eval_on_single_gpu(model, valset):
    model.eval()
    loader = dataloader.DataLoader(valset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers)
    generate_results = []
    
    with torch.no_grad():
        for data in tqdm(loader, mininterval=60, disable=False):
            if args.dataset == "spot":
                bef_imgs = data['bef_img'].cuda()
                aft_imgs = data['aft_img'].cuda()
                img_ids = data['img_id']
                ref_tokens = data['lexicon']
                captions = model.generate(bef_imgs, aft_imgs, ref_tokens)

                for i, (img_id, caption) in enumerate(zip(img_ids, captions)):
                    cap_item = {"image_id":img_id, "caption":caption}
                    generate_results.append(cap_item)
            
            elif args.dataset == "clevr":
                bef_imgs = data['bef_image'].cuda()
                aft_imgs = data['aft_image'].cuda()
                nsc_imgs = data['nsc_image'].cuda()
                img_ids = data['img_id']
                sc_ref_tokens = data['sc_ref_words']
                nsc_ref_tokens = data['nsc_ref_words']

                sc_captions = model.generate(bef_imgs, aft_imgs, ref_tokens=sc_ref_tokens)
                nsc_captions = model.generate(bef_imgs, nsc_imgs, ref_tokens=nsc_ref_tokens) 
                for i, (img_id, caption) in enumerate(zip(img_ids, sc_captions)):
                    cap_item = {"image_id":img_id, "caption":caption}
                    generate_results.append(cap_item)

                for i, (img_id, caption) in enumerate(zip(img_ids, nsc_captions)):
                    cap_item = {"image_id": "%s_n" % img_id, "caption":caption}
                    generate_results.append(cap_item)

            elif args.dataset == "IER":
                for data in tqdm(loader, disable=False):
                    bef_imgs = data['bef_img'].cuda()
                    aft_imgs = data['aft_img'].cuda()
                    img_ids = data['img_id']
                    ref_tokens = data['lexicon']
                    captions = model.generate(bef_imgs, aft_imgs, ref_tokens, max_length=args.max_length)
                    for i, (img_id, caption) in enumerate(zip(img_ids, captions)):
                        cap_item = {"image_id":img_id, "caption":caption}
                        generate_results.append(cap_item)

    json.dump(generate_results, open(os.path.join(args.model_dir, 'val_total_captions.json'), 'w'))

    if args.dataset == 'clevr':
        gt_file = os.path.join(args.gt_dir, "%s_test_change_captions_reformat.json" % args.dataset)
    else:
        gt_file = os.path.join(args.gt_dir, "%s_val_change_captions_reformat.json" % args.dataset)
    with utils.HiddenPrints():
        generation_metrics = utils.generation_score(gt_file, os.path.join(args.model_dir, 'val_total_captions.json'))

    return generation_metrics


def test(model, valset):
    model.eval()
    loader = dataloader.DataLoader(valset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers)
    generate_results = []
    sc_generate_results = []
    nsc_generate_results = []
    with torch.no_grad():
        for data in tqdm(loader, mininterval=60, disable=False):
            if args.dataset == "spot":
                bef_imgs = data['bef_img'].cuda()
                aft_imgs = data['aft_img'].cuda()
                img_ids = data['img_id']
                ref_tokens = data['lexicon']
                captions = model.generate(bef_imgs, aft_imgs, ref_tokens)

                for i, (img_id, caption) in enumerate(zip(img_ids, captions)):
                    cap_item = {"image_id":img_id, "caption":caption}
                    generate_results.append(cap_item)
            
            elif args.dataset == "clevr":
                bef_imgs = data['bef_image'].cuda()
                aft_imgs = data['aft_image'].cuda()
                nsc_imgs = data['nsc_image'].cuda()
                img_ids = data['img_id']
                sc_ref_tokens = data['sc_ref_words']
                nsc_ref_tokens = data['nsc_ref_words']

                sc_captions = model.generate(bef_imgs, aft_imgs, ref_tokens=sc_ref_tokens)
                nsc_captions = model.generate(bef_imgs, nsc_imgs, ref_tokens=nsc_ref_tokens)
                
                for i, (img_id, caption) in enumerate(zip(img_ids, sc_captions)):
                    cap_item = {"image_id":img_id, "caption":caption}
                    generate_results.append(cap_item)
                    sc_generate_results.append(cap_item)

                for i, (img_id, caption) in enumerate(zip(img_ids, nsc_captions)):
                    cap_item = {"image_id": "%s_n" % img_id, "caption":caption}
                    generate_results.append(cap_item)
                    nsc_generate_results.append(cap_item)
            
            elif args.dataset == "IER":
                for data in tqdm(loader, disable=False):
                    bef_imgs = data['bef_img'].cuda()
                    aft_imgs = data['aft_img'].cuda()
                    img_ids = data['img_id']
                    ref_tokens = data['lexicon']
                    captions = model.generate(bef_imgs, aft_imgs, ref_tokens, max_length=args.max_length)
                    for i, (img_id, caption) in enumerate(zip(img_ids, captions)):
                        cap_item = {"image_id":img_id, "caption":caption}
                        generate_results.append(cap_item)

    json.dump(generate_results, open(os.path.join(args.model_dir, 'total_captions.json'), 'w'))

    with utils.HiddenPrints():
        generation_metrics = utils.generation_score(os.path.join(args.gt_dir, "%s_test_change_captions_reformat.json" % args.dataset), os.path.join(args.model_dir, 'total_captions.json'))
   
    logging.info(generation_metrics)
    return generation_metrics
        

def save_model(model, epoch):
    save_file_path = os.path.join(args.model_dir, "%s_model_params.pth"%(args.dataset))
    state_dict = model.state_dict()

    torch.save(state_dict, save_file_path)
    return save_file_path


if __name__ == '__main__':

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    logging.info('save arguments...')
    for k in args.__dict__.keys():
        logging.info("\t'{}'={}".format(k, args.__dict__[k]))  

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('Loading the datasets and model...')

    if args.mode == 'train':
        logging.info('Loading the datasets and model...')

        train_set = get_dataset(args.dataset, split='train')
        val_set = get_dataset(args.dataset, split='val')

        model, optimizer = create_model_and_optimizer()
        train_and_evaluate(model, optimizer, train_set, val_set)

    elif args.mode == 'test':
        test_set = get_dataset(args.dataset, split='test')
        model = create_model_and_optimizer(need_optim=False)
        model.load_state_dict(torch.load(args.model_pth), strict=False)
        logging.info("eval on test set....")
        test(model, test_set)
        