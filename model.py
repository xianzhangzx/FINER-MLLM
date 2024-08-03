import torch
import yaml
import numpy as np
import torch.nn.functional as F
from torch import nn
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train, LayerNorm

from LoraViT import create_lora_eva_vit_g
from LoraQformer import BertConfig, BertLMHeadModel
import loralib as lora
from transformers.models.llama.configuration_llama import LlamaConfig
from llm_model import AssistantVicuna
from transformers import LlamaTokenizer
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
from timm.models.layers import trunc_normal_

PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/blip2_instruct_vicuna13b.yaml",
    }


class FINER_MLLM(Blip2Base):
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        qformer_text_input=False,
        apply_lora_for_qformer=True,
        consist_w=1.0,
        ortho_w=0.1,
        qformer_lora_k=4,
        vit_lora_k=4
    ):
        super().__init__()    
        
        self.consist_w = consist_w
        self.ortho_w = ortho_w
        self.qformer_lora_k = qformer_lora_k
        self.vit_lora_k = vit_lora_k

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            lora.mark_only_lora_as_trainable(self.visual_encoder)
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

        if apply_lora_for_qformer:
            self.Qformer, self.query_tokens = self.init_LoraQformer(
                num_query_token, self.visual_encoder.num_features, lora_k=self.qformer_lora_k
            )
            lora.mark_only_lora_as_trainable(self.Qformer)
        else:
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features,
            )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        # 0:unk 1:bos 2:eos pad_token is none; length of tokenizer is 32000
        self.llm_tokenizer.add_special_tokens({'pad_token': '</s>'})
        self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token
        
        llm_config = LlamaConfig.from_pretrained(llm_model)
        self.llm_model = AssistantVicuna(llm_model, llm_config)
        
        self.llm_model.config.pad_token_id = 0
        self.llm_model.config.bos_token_id = 1
        self.llm_model.config.eos_token_id = 2

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        
        self.qformer_text_input = qformer_text_input

    def forward(self, bef_image, aft_image, caption, ref_tokens=None, ref_texts=None):
        device = bef_image.device
        with self.maybe_autocast():
            bef_image_embeds = self.ln_vision(self.visual_encoder(bef_image))
            aft_image_embeds = self.ln_vision(self.visual_encoder(aft_image))

            bef_image_atts = torch.ones(bef_image_embeds.size()[:-1], dtype=torch.long).to(device)
            aft_image_atts = torch.ones(aft_image_embeds.size()[:-1], dtype=torch.long).to(device)

            batch_size = bef_image.size(0)
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    [self.prompt]*batch_size,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt"
                ).to(bef_image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(bef_image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
                bef_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=bef_image_embeds,
                        encoder_attention_mask=bef_image_atts,
                        return_dict=True,
                        output_attentions=True
                )
                aft_query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=aft_image_embeds,
                    encoder_attention_mask=aft_image_atts,
                    return_dict=True,
                    output_attentions=True
                )
            else:
                bef_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=bef_image_embeds,
                        encoder_attention_mask=bef_image_atts,
                        return_dict=True,
                        output_attentions=True
                )
                aft_query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=aft_image_embeds,
                    encoder_attention_mask=aft_image_atts,
                    return_dict=True,
                    output_attentions=True
                )

            inputs_llm = torch.cat([self.llm_proj(bef_query_output.last_hidden_state[:, :query_tokens.size(1), :]), 
                                    self.llm_proj(aft_query_output.last_hidden_state[:, :query_tokens.size(1), :])], dim=1)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            [self.prompt]*bef_image.shape[0],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in caption],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.backbone.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
        ref_token_ids, ref_attention_mask = self.get_token_ids(ref_tokens)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                decoder_ref_token_ids=ref_token_ids,
                decoder_ref_attention_mask=ref_attention_mask,
            )

        consistency_loss, orthogonal_loss = self.attention_consistency_loss(bef_query_output.cross_attentions[-2], aft_query_output.cross_attentions[-2]) # [32, 12, 32, 257]
        loss = outputs.loss + self.consist_w * consistency_loss + self.ortho_w * orthogonal_loss
        return loss

    @torch.no_grad()
    def generate(
        self,
        bef_image,
        aft_image,
        ref_tokens=None,
        word_list=None,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=50,
        min_length=5,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        prompt = self.prompt
        bs = bef_image.size(0)
        device = bef_image.device

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        query_tokens = self.query_tokens.expand(bs, -1, -1)

        with self.maybe_autocast():
            bef_image_embeds = self.ln_vision(self.visual_encoder(bef_image))
            aft_image_embeds = self.ln_vision(self.visual_encoder(aft_image))

            bef_image_atts = torch.ones(bef_image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            aft_image_atts = torch.ones(aft_image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    prompt,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt"
                ).to(bef_image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(bef_image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
                bef_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=bef_image_embeds,
                        encoder_attention_mask=bef_image_atts,
                        return_dict=True,
                        output_attentions=True
                )
                aft_query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=aft_image_embeds,
                    encoder_attention_mask=aft_image_atts,
                    return_dict=True,
                    output_attentions=True
                )
            else:
                bef_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=bef_image_embeds,
                        encoder_attention_mask=bef_image_atts,
                        return_dict=True,
                        output_attentions=True
                )
                aft_query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=aft_image_embeds,
                    encoder_attention_mask=aft_image_atts,
                    return_dict=True,
                    output_attentions=True
                )
        
            inputs_llm = torch.cat([self.llm_proj(bef_query_output.last_hidden_state[:, :query_tokens.size(1), :]), 
                                    self.llm_proj(aft_query_output.last_hidden_state[:, :query_tokens.size(1), :])], dim=1)

            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(self.device)
        
        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(device)
        prompt_length = llm_tokens.attention_mask.shape[1]

        if word_list is not None:
            word_list_ids = self.llm_tokenizer.convert_tokens_to_ids(self.llm_tokenizer.tokenize(word_list)) + [self.llm_tokenizer.unk_token_id, self.llm_tokenizer.eos_token_id, self.llm_tokenizer.bos_token_id]
            bad_words_ids = self.get_bad_words_ids(word_list_ids)
        else:
            bad_words_ids = None

        ref_token_ids, ref_attention_mask = self.get_token_ids(ref_tokens)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.backbone.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_ref_token_ids=ref_token_ids,
                decoder_ref_attention_mask=ref_attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                bad_words_ids=bad_words_ids
            )

        output_text = self.llm_tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    
    @classmethod
    def load_pretrained_model_from_blip2(cls, model_type, qformer_lora_k=4, vit_lora_k=4, consist_w=1.0, ortho_w=0.1):
        cfg_path = PRETRAINED_MODEL_CONFIG_DICT[model_type]
        with open(cfg_path, 'r', encoding='utf-8') as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = model_cfg['model']
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            consist_w=consist_w,
            ortho_w=ortho_w,
            qformer_lora_k = qformer_lora_k,
            vit_lora_k = vit_lora_k
        )

        model.load_checkpoint_from_config(cfg)

        return model

    def attention_consistency_loss(self, bef_attention_score, aft_attention_score):
        # attention_score: (B, H, L, 514)
        num_patchs = self.visual_encoder.patch_embed.num_patches + 1
        bef_attention_score = bef_attention_score[:, :, :, 1:num_patchs] # (B, H, L, 256)
        aft_attention_score = aft_attention_score[:, :, :, 1:num_patchs] # (B, H, L, 256)
        consistency_loss = (self.kl_loss(bef_attention_score, aft_attention_score) + self.kl_loss(aft_attention_score, bef_attention_score)) / 2
        orthogonal_loss = (self.attention_orthogonal_regularization(bef_attention_score) + self.attention_orthogonal_regularization(aft_attention_score)) / 2
        return consistency_loss, orthogonal_loss

    def kl_loss(self, x, y):
        log_soft_x = F.log_softmax(x / 0.1, dim=-1)
        soft_y = F.softmax(y / 0.05, dim=-1)
        return F.kl_div(log_soft_x, soft_y)
    
    def attention_orthogonal_regularization(self, attn):
        batch_size, head, k, length = attn.shape
        attn = F.softmax(attn, dim=-1)
        attn = attn.reshape(-1, k, length) # (N,k,L)
        cosine_score = torch.matmul(attn, attn.permute(0,2,1).contiguous())
        soft_logits = - F.log_softmax(cosine_score / 0.05, dim=-1)
        loss = soft_logits.diagonal(dim1=1, dim2=2).mean()
        return loss

    def init_LoraQformer(cls, num_query_token, vision_width, cross_attention_freq=2, apply_lora=True, lora_k=4):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.apply_lora = apply_lora
        encoder_config.lora_r = lora_k
        # encoder_config.lora_alpha = lora_k * 2
        encoder_config.lora_alpha = 1
        encoder_config.apply_adapter = False
        # encoder_config.adapter_size = 64

        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_vision_encoder(self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision):
        visual_encoder = create_lora_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision, lora_r=self.vit_lora_k
        )
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def get_bad_words_ids(self, valid_words_list):
        bad_words_ids = []
        for i in range(self.llm_tokenizer.vocab_size):
            if i not in valid_words_list:
                bad_words_ids.append(i)
        # print(len(bad_words_ids), len(valid_words_list))
        return [bad_words_ids]

    def get_token_ids(self, bag_of_words):
        assert type(bag_of_words) == list and type(bag_of_words[0]) == str, "error format of ref tokens"
        max_length = 0
        all_bag_tokens = []
        # self.tokenizer.pad_
        for bag in bag_of_words:
            tokens = self.llm_tokenizer.tokenize(bag)
            length = len(tokens)
            max_length = max(max_length, length)
            all_bag_tokens.append((tokens, length))

        input_ids = torch.zeros(len(all_bag_tokens), max_length).long()
        attention_mask = torch.zeros(len(all_bag_tokens), max_length).long()

        for i, (tokens, length) in enumerate(all_bag_tokens):
            input_ids[i, :length] = torch.LongTensor(self.llm_tokenizer.convert_tokens_to_ids(tokens))
            attention_mask[i, :length] = 1

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        return input_ids, attention_mask
    
