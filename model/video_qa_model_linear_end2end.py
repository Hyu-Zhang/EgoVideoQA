# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb
import math
import timm
import torch
import yaml
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel
from einops import rearrange, repeat

from base import BaseModel
# from model import video_transformer
from model.video_transformer import SpaceTimeTransformer
from utils.util import state_dict_data_parallel_fix

from model import roberta
from model.roberta import RobertaModel, _prepare_decoder_attention_mask
from model import heads
from transformers import RobertaConfig
from functools import partial
import copy
import torch.distributed as dist

with open('./EgoNCE_MLM_ITM_Config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class FrozenInTime(BaseModel):
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=4096,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='bilinear', #bilinear
                 config = config,
                 task_names = 'EgoNCE_ITM_MLM',
                 norm_layer = None,
                 embed_dim=768,
                 model_dim=768,
                 output_dim=1010):
                 #egotaskqa_head_output_dim=1010):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        self.config = config
        #self.egotaskqa_head_output_dim = egotaskqa_head_output_dim
        self.task_names = task_names
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        if self.text_params['model'].startswith('roberta'):
            self.text_model = RobertaModel.from_pretrained("/userhome/pretrain_model/roberta-base", local_files_only=True)
        self.text_model.train()

        pretrained = video_params['pretrained']
        if video_params['model'] == "SpaceTimeTransformer":
            self.num_frames = 16 #self.config["data_loader"][0]["args"]["video_params"]["num_frames"]
            time_init = 'zeros'
            attention_style = 'frozen-in-time'
            arch_config = 'base_patch16_224'
            vit_init = 'imagenet-21k'
            if arch_config == 'base_patch16_224':
                #vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                vit_model = torch.load("/userhome/pretrain_model/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                model = SpaceTimeTransformer(num_frames=self.num_frames,
                                            time_init=time_init,
                                            attention_style=attention_style)
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
           
            if load_checkpoint in ["", None]:
                vit_checkpoint = vit_model
                new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint, model.state_dict())
                model.load_state_dict(new_vit_dict, strict=False)
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        self.alpha = nn.Parameter(torch.Tensor([0]))
        self.belta = nn.Parameter(torch.Tensor([0]))
        # Project to a common embedding
        if projection == 'minimal':

            txt_proj = nn.Sequential(
                                    nn.Linear(self.text_model.config.hidden_size, projection_dim, bias=False),
                                    nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
                                    nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
                                    )

            vid_proj = nn.Sequential(
                                    nn.Linear(ftr_dim, projection_dim, bias=False),
                                    nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
                                    nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
                                    )
        
        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj


        if ('MLM' in self.task_names or 'ITM' in self.task_names):
            # for FIBER-like cross-attention

            bert_config = RobertaConfig(
                vocab_size=self.config["vocab_size"],
                hidden_size=self.config["hidden_size"],
                num_hidden_layers=self.config["num_layers"],
                num_attention_heads=self.config["num_heads"],
                intermediate_size=self.config["hidden_size"] * config["mlp_ratio"],
                #max_position_embeddings=maxlen, [was used in BTGOT script]
                hidden_dropout_prob=self.config["drop_rate"],
                attention_probs_dropout_prob=self.config["drop_rate"],
            )

            self.num_fuse_block=self.config["num_fuse_block"]
            self.num_text_layer=self.config["num_layers"]
            roberta.NUM_FUSE_BLOCK = self.video_model.NUM_FUSE_BLOCK=self.num_fuse_block
            roberta.DIM_IMG=self.config["input_image_embed_size"]
            self.video_model.DIM_TXT=self.config["input_text_embed_size"]

            self.cross_modal_text_transform = nn.Linear(self.config["input_text_embed_size"], self.config["hidden_size"])
            self.cross_modal_text_transform.apply(init_weights)
            self.cross_modal_video_transform = nn.Linear(self.config["input_image_embed_size"], self.config["hidden_size"])
            self.cross_modal_video_transform.apply(init_weights)

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            self.num_patches = self.video_model.patch_embed.num_patches
            self.patches_per_frame = self.num_patches//self.num_frames
            norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
            self.norm = norm_layer(embed_dim)
            self.pre_logits = nn.Identity()

            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.cross_modal_video_pooler = heads.Pooler(config["hidden_size"])
            self.cross_modal_video_pooler.apply(init_weights)
            self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
            self.cross_modal_text_pooler.apply(init_weights)

            self.dropout = nn.Dropout(p=0.2)
            self.relu = nn.ReLU()
            self.projector_1 = nn.Linear(model_dim, output_dim)
            self.projector_2 = nn.Linear(output_dim, output_dim)

            ## einops transformations
            self.einops_from_space = 'b (f n) d'
            self.einops_to_space = '(b f) n d'
            self.einops_from_time = 'b (f n) d'
            self.einops_to_time = '(b n) f d'

        if 'MLM' in self.task_names:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(init_weights)

        if 'ITM' in self.task_names:
            self.itm_score = heads.ITMHead(config["hidden_size"] * 2)
            self.itm_score.apply(init_weights)

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint, map_location='cpu')
            print("This loop is being executed")
            # original
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=False)
            print("Model is loaded with pre-trained parameters")
        
    def set_device(self, device):
        self.device = device
        
    def up_sampling(self, large_patch_emb, b, curr_frames, num):
        cls_patch, large_out_emb = large_patch_emb[:, 0, :], large_patch_emb[:, 1:, :]
        large_out_emb = large_out_emb.contiguous().view(b*curr_frames, num, num, self.video_model.patch_embed.embed_dim)
        large_out_emb = large_out_emb.repeat_interleave(2, 2).repeat_interleave(2, 1).view(b, -1, self.video_model.patch_embed.embed_dim)
        large_out_emb = torch.cat([cls_patch.unsqueeze(1), large_out_emb], dim=1)
        return large_out_emb
    
    def down_sampling(self, video_data_itm, b, curr_frames, num, flag=0):
        cls_patch, large_patch_emb = video_data_itm[:, 0, :], video_data_itm[:, 1:, :]
        if flag == 0:
            large_patch_emb = large_patch_emb.view(b, curr_frames, num, num, self.video_model.patch_embed.embed_dim)
            large_patch_emb = F.max_pool3d(large_patch_emb, kernel_size=(2, 2, 1), stride=(2, 2, 1))
            large_patch_emb = large_patch_emb.view(b, curr_frames*(num//2)*(num//2), self.video_model.patch_embed.embed_dim)
        elif flag == 1:
            large_patch_emb = large_patch_emb.view(b, curr_frames, num, num, self.video_model.patch_embed.embed_dim)
            large_patch_emb = F.max_pool3d(large_patch_emb, kernel_size=(num, num, 1), stride=(num, num, 1))
            large_patch_emb = large_patch_emb.view(b, curr_frames, self.video_model.patch_embed.embed_dim)
        elif flag == 2:
            large_patch_emb = large_patch_emb.view(b, curr_frames, self.video_model.patch_embed.embed_dim)
            large_patch_emb = F.max_pool2d(large_patch_emb, kernel_size=(curr_frames, 1), stride=(curr_frames, 1))
            large_patch_emb = large_patch_emb.view(b, 1, self.video_model.patch_embed.embed_dim)
        large_patch_emb = torch.cat([cls_patch.unsqueeze(1), large_patch_emb], dim=1)
        return large_patch_emb
    
    def mask_down_sampling(self, video_data_itm, b, curr_frames, num, flag=0):
        cls_patch, large_patch_emb = video_data_itm[:, 0], video_data_itm[:, 1:]
        if flag == 0:
            large_patch_emb = large_patch_emb.view(b, curr_frames, num, num)
            large_patch_emb = F.max_pool2d(large_patch_emb, kernel_size=(2, 2), stride=(2, 2))
            large_patch_emb = large_patch_emb.view(b, curr_frames*(num//2)*(num//2))
        elif flag == 1:
            large_patch_emb = large_patch_emb.view(b, curr_frames, num, num)
            large_patch_emb = F.max_pool2d(large_patch_emb, kernel_size=(num, num), stride=(num, num))
            large_patch_emb = large_patch_emb.view(b, curr_frames)
        elif flag == 2:
            large_patch_emb = large_patch_emb.view(b, curr_frames)
            large_patch_emb = F.max_pool1d(large_patch_emb, kernel_size=(curr_frames,), stride=(curr_frames,))
            large_patch_emb = large_patch_emb.view(b, 1)
        large_patch_emb = torch.cat([cls_patch.unsqueeze(1), large_patch_emb], dim=1)
        return large_patch_emb
    
    def forward(self, data, video_only=False, return_embeds=True):
        
        text_data = data['text']
        video_data = data['video'] # torch.Size([48, 16, 3, 224, 224])

        b, curr_frames, channels, _, _ = video_data.shape
        video_data_itm = self.video_model.patch_embed(video_data) # torch.Size([768, 768, 14, 14])
        video_data_itm = video_data_itm.flatten(2).transpose(2, 1) # torch.Size([768, 196, 768])
        video_data_itm = video_data_itm.reshape(b, -1, self.video_model.patch_embed.embed_dim) #torch.Size([48, 3136, 768])

        # video position embedding
        BF = video_data_itm.shape[0]
        cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # torch.Size([48, 1, 768])
        video_data_itm = torch.cat((cls_tokens, video_data_itm), dim=1)
        # torch.Size([48, 3137, 768])
        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.video_model.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.video_model.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
    
        tile_large_pos_embed = self.video_model.large_pos_embed.view(1, 7, 7, self.video_model.patch_embed.embed_dim)
        tile_large_pos_embed = tile_large_pos_embed.repeat_interleave(2, 2).repeat_interleave(2, 1)
        tile_large_pos_embed = tile_large_pos_embed.view(1, -1, self.video_model.patch_embed.embed_dim).repeat(1, self.num_frames, 1)
        
        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = self.video_model.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
        total_pos_embed = tile_pos_embed + tile_large_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        f = curr_frames
        unfused_blocks = self.num_text_layer - self.num_fuse_block
        curr_patches = video_data_itm.shape[1]
        video_data_itm = video_data_itm + total_pos_embed[:, :curr_patches]
        video_data_itm = self.video_model.pos_drop(video_data_itm)
        
        n = self.patches_per_frame
        sqrt_n = int(math.sqrt(n))
        mask_patch = None
        # down sampling
        large_patch_emb = self.down_sampling(video_data_itm, b, curr_frames, sqrt_n)
        # encode video
        for blk_i, (cloned_blk, blk) in enumerate(zip(self.video_model.cloned_blocks[:unfused_blocks], self.video_model.blocks[:unfused_blocks])):
            if config['use_checkpoint']:
                # encode patch
                large_patch_emb, score_big = torch.utils.checkpoint.checkpoint(cloned_blk, large_patch_emb, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, n//4, f, None)
                # encode sub-patch
                video_data_itm, score_sub = torch.utils.checkpoint.checkpoint(blk, video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, n, f, None)
            else:
                large_patch_emb, score_big = cloned_blk(large_patch_emb, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                time_n=n//4, space_f=f, patch_mask=None)
                video_data_itm, score_sub = blk(video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                time_n=n, space_f=f, patch_mask=mask_patch)
        
            # up sampling
            large_out_emb_up = self.up_sampling(large_patch_emb, b, curr_frames, sqrt_n//2)
            # down sampling
            video_data_itm_down = self.down_sampling(video_data_itm, b, curr_frames, sqrt_n)
            # fusion
            video_data_itm = video_data_itm + self.alpha * large_out_emb_up
            large_patch_emb = large_patch_emb + self.belta * video_data_itm_down
            
#             if (blk_i + 1)%6 == 0:
                # merge score
        score_big = score_big.view(b*curr_frames, sqrt_n//2, sqrt_n//2)
        score_big = score_big.repeat_interleave(2, 2).repeat_interleave(2, 1).view(b, curr_frames, -1)
        score = score_sub.softmax(dim=-1) + score_big.softmax(dim=-1)
        topn = (sqrt_n - 2) * (sqrt_n - 2)
        # patch selection
        mask_patch = self.video_model.patch_selection(score, topn)
        
        # encode text
        text_embeds = self.text_model.embeddings(input_ids=text_data['input_ids']) # before it was input_ids=text_ids
        device = text_embeds.device
        text_masks = text_data['attention_mask']
        input_shape = text_masks.size()
        extend_text_masks = self.text_model.get_extended_attention_mask(text_masks, input_shape, device)
        for layer_i, layer in enumerate(self.text_model.encoder.layer[:unfused_blocks]):
            if config['use_checkpoint']:
                text_embeds = torch.utils.checkpoint.checkpoint(layer, text_embeds, extend_text_masks)[0]
            else:
                text_embeds = layer(text_embeds, extend_text_masks)[0]
                
        # cross-modal interaction
        for blk_i, blk in enumerate(self.video_model.blocks[unfused_blocks:self.num_text_layer]):
            if blk_i == 3:
                video_data_itm = self.down_sampling(video_data_itm, b, curr_frames, sqrt_n)
                mask_patch = self.mask_down_sampling(mask_patch, b, curr_frames, sqrt_n)
                n = n//4
                sqrt_n = sqrt_n//2
            if blk_i == 4:
                video_data_itm = self.down_sampling(video_data_itm, b, curr_frames, sqrt_n, flag=1)
                mask_patch = self.mask_down_sampling(mask_patch, b, curr_frames, sqrt_n, flag=1)
                n = 1
            if blk_i == 5:
                video_data_itm = self.down_sampling(video_data_itm, b, curr_frames, sqrt_n, flag=2)
                mask_patch = self.mask_down_sampling(mask_patch, b, curr_frames, sqrt_n, flag=2)
                n = 1
                f = 1
            
            if config['use_checkpoint']:
                fuse_video_data, score = torch.utils.checkpoint.checkpoint(blk, video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, n, f, mask_patch, text_embeds, extend_text_masks)
                text_embeds = torch.utils.checkpoint.checkpoint(self.text_model.encoder.layer[blk_i + unfused_blocks],
                                          text_embeds, extend_text_masks, None, (video_data_itm), None, None, False, True)[0]
            else:
                fuse_video_data, score = blk(video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, 
                                          y=text_embeds, y_mask=extend_text_masks, time_n=n, space_f=f, patch_mask=mask_patch)
                text_embeds = self.text_model.encoder.layer[blk_i + unfused_blocks](text_embeds, extend_text_masks, encoder_hidden_states=(video_data_itm), last_norm=True)[0]
            video_data_itm = fuse_video_data


        video_data_itm = self.norm(video_data_itm)[:, :]
        video_data_itm = self.pre_logits(video_data_itm)
        
        video_cls = video_data_itm[:,0,:]
#         video_cls = video_data_itm.mean(dim=1)
        batch_out = self.projector_2(self.dropout(self.relu(self.projector_1(video_cls))))

        return batch_out

    
    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        elif self.text_params['model'].startswith('roberta'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_text_tokens(self, text_data, is_proj=True):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']    # not implement for bert
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        elif self.text_params['model'].startswith('roberta'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        else:
            raise NotImplementedError
        
        if is_proj:
            text_embeddings = self.txt_proj(text_embeddings)
        
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode, align_corners=True).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def sim_matrix_batch_val(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1).unsqueeze(-1), b.norm(dim=-1).unsqueeze(-1)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim_mt


if __name__ == "__main__":
    video_params = {"model": SpaceTimeTransformer, "arch_config": "base_patch16_224", "num_frames": 4, "pretrained": True, "time_init": "zeros"}
    text_params: {"model": "roberta-base", "pretrained": True, "input": "text"}
    model = FrozenInTime(video_params, text_params)
