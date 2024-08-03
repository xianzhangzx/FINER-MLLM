import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import replace_return_docstrings
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss

from lavis.models.blip2_models.modeling_llama import LlamaPreTrainedModel, LlamaForCausalLM, LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.utils import add_start_docstrings_to_model_forward

class AssistantGenerator(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, hidden_size=768):
        super().__init__()
        self.sub_bag_weight = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, dropout=0.3, kdim=hidden_size, vdim=hidden_size, batch_first=True)
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size 
        self.vocab_size = vocab_size 
        self.ln = nn.LayerNorm(hidden_size)
        # for name, param in self.sub_bag_weight.named_parameters():
        #     nn.init.zeros_(param)

    def forward(self, ref_token_ids, ref_token_embeds=None, ref_attention_mask=None, hidden_states=None):
        """
            hidden_states: output of last_layer in bert-decoder
            ref_token_embeds: the embeding of retrieval tokens
            ref_attention_mask: attention mask for pad tokens
        """
        total_batch_size, input_length = hidden_states.shape[:2] # (B*beam_num, L)
        batch_size, ref_token_length = ref_attention_mask.shape # (B,P)

        if total_batch_size != batch_size:
            assert (total_batch_size % batch_size) == 0, 'error, total_batch_size must be num_beams * batch_size during the generation process.'
            num_beams = int(total_batch_size // batch_size)
            ref_token_ids = ref_token_ids.repeat_interleave(num_beams, dim=0)
            ref_token_embeds = ref_token_embeds.repeat_interleave(num_beams, dim=0)
            ref_attention_mask = ref_attention_mask.repeat_interleave(num_beams, dim=0)

        # ref_token_embeds: (B*beam_num, sentece_L, embed_size)
        ref_token_embeds = ref_token_embeds.repeat_interleave(input_length, dim=0) # (B,P,D)=>(BL,P,embed_size)
        ref_attention_mask = ref_attention_mask.repeat_interleave(input_length, dim=0) # (BL, P)
        hidden_states = hidden_states.reshape(-1, self.hidden_size).unsqueeze(1) # (BL,1,hidden_size)

        # single-head cross-attention
        cross_attn_output, cross_attention_weights = self.sub_bag_weight(hidden_states, ref_token_embeds, ref_token_embeds, key_padding_mask=~(ref_attention_mask.bool()), need_weights=True)

        expanded_sub_wieght = torch.zeros(total_batch_size*input_length, self.vocab_size).to(hidden_states.device)
        ind = ref_token_ids.repeat_interleave(input_length, dim=0).long()
        # expanded_sub_wieght:(BL, vocab_size)  ind:(BL,P)  cross_attn_w:(BL,P)
        expanded_sub_wieght = expanded_sub_wieght.scatter_(1, ind, cross_attention_weights.squeeze()).reshape(total_batch_size, input_length, self.vocab_size)
        return expanded_sub_wieght


class AssistantVicuna(LlamaPreTrainedModel):
    def __init__(self, llm_model, config):
        super().__init__(config)
        self.backbone = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16, config=config
        )
        # print(self.backbone.config)
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.assistant = AssistantGenerator(vocab_size=config.vocab_size, hidden_size=config.hidden_size)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_ref_token_ids: torch.LongTensor = None,
        decoder_ref_attention_mask: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        reduction: Optional[str] = "mean",
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.backbone.lm_head(hidden_states)
        ref_token_embeds = self.backbone.get_input_embeddings()(decoder_ref_token_ids)
        assist_weight = self.assistant(decoder_ref_token_ids, ref_token_embeds, decoder_ref_attention_mask, hidden_states.detach())
        base_weight = torch.ones_like(logits).to(self.device)
        logits = logits * (base_weight + assist_weight)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction=reduction)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if reduction == "none":
                loss = loss.view(logits.size(0), -1).mean(1)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "decoder_ref_token_ids": kwargs.get("decoder_ref_token_ids", None),
                "decoder_ref_attention_mask": kwargs.get("decoder_ref_attention_mask", None)
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

