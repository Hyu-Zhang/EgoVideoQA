import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.nn.modules.module import Module
from torch.nn import functional as F

class CRN(Module):
    def __init__(self, module_dim, num_objects, max_subset_size, gating=False, spl_resolution=1):
        super(CRN, self).__init__()
        self.module_dim = module_dim
        self.gating = gating

        self.k_objects_fusion = nn.ModuleList()
        if self.gating:
            self.gate_k_objects_fusion = nn.ModuleList()
        for i in range(min(num_objects, max_subset_size + 1), 1, -1):
            self.k_objects_fusion.append(nn.Linear(2 * module_dim, module_dim))
            if self.gating:
                self.gate_k_objects_fusion.append(nn.Linear(2 * module_dim, module_dim))
        self.spl_resolution = spl_resolution
        self.activation = nn.ELU()
        self.max_subset_size = max_subset_size

    def forward(self, object_list, cond_feat):
        """
        :param object_list: list of tensors or vectors
        :param cond_feat: conditioning feature
        :return: list of output objects
        """
        scales = [i for i in range(len(object_list), 1, -1)]

        relations_scales = []
        subsample_scales = []
        for scale in scales:
            relations_scale = self.relationset(len(object_list), scale)
            relations_scales.append(relations_scale)
            subsample_scales.append(min(self.spl_resolution, len(relations_scale)))

        crn_feats = []
        if len(scales) > 1 and self.max_subset_size == len(object_list):
            start_scale = 1
        else:
            start_scale = 0
        for scaleID in range(start_scale, min(len(scales), self.max_subset_size)):
            idx_relations_randomsample = np.random.choice(len(relations_scales[scaleID]),
                                                          subsample_scales[scaleID], replace=False)
            mono_scale_features = 0
            for id_choice, idx in enumerate(idx_relations_randomsample):
                clipFeatList = [object_list[obj].unsqueeze(1) for obj in relations_scales[scaleID][idx]]
                # g_theta
                g_feat = torch.cat(clipFeatList, dim=1)
                g_feat = g_feat.mean(1)
                if len(g_feat.size()) == 2:
                    h_feat = torch.cat((g_feat, cond_feat), dim=-1)
                elif len(g_feat.size()) == 3:
                    cond_feat_repeat = cond_feat.repeat(1, g_feat.size(1), 1)
                    h_feat = torch.cat((g_feat, cond_feat_repeat), dim=-1)
                if self.gating:
                    h_feat = self.activation(self.k_objects_fusion[scaleID](h_feat)) * torch.sigmoid(
                        self.gate_k_objects_fusion[scaleID](h_feat))
                else:
                    h_feat = self.activation(self.k_objects_fusion[scaleID](h_feat))
                mono_scale_features += h_feat
            crn_feats.append(mono_scale_features / len(idx_relations_randomsample))
        return crn_feats

    def relationset(self, num_objects, num_object_relation):
        return list(itertools.combinations([i for i in range(num_objects)], num_object_relation))

class Interaction(nn.Module):
    def __init__(self, k_max_frame_level=25, k_max_clip_level=4, spl_resolution=1, vision_dim=32, module_dim=32):
        super(Interaction, self).__init__()

        # self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=True, spl_resolution=spl_resolution)
        # self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=True, spl_resolution=spl_resolution)

        # self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        # self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        # self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        # self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()
        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

    def forward(self, appearance_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size,
             num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        cls_emb, patch_emb = appearance_video_feat[:, 0, :], appearance_video_feat[:, 1:, :]
        
        batch_size = appearance_video_feat.size(0)
        patch_emb = patch_emb.view(batch_size, 16, 100, 768)
        
        clip_level_crn_outputs = []
        # question_embedding_proj = self.question_embedding_proj(question_embedding)
        for i in range(patch_emb.size(1)):#4

            clip_level_appearance = patch_emb[:, i, :, :]  # (bz, 100, 768)
            clip_level_crn_question = self.clip_level_question_cond(torch.unbind(clip_level_appearance, dim=1), question_embedding)

            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        # Encode video level motion
        # _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        # video_level_motion = video_level_motion.transpose(0, 1)
        # video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # # video level CRNs
        # video_level_crn_motion = self.video_level_motion_cond(clip_level_crn_outputs, video_level_motion_feat_proj)
        video_level_crn_question = self.video_level_question_cond(clip_level_crn_outputs,
                                                                  question_embedding.unsqueeze(1))

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        v_q_cat = torch.cat((video_level_crn_output, question_embedding.unsqueeze(1) * video_level_crn_output), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * video_level_crn_output).sum(1)
        
        output = cls_emb + v_distill

        return output
    

if __name__ == "__main__":
    x = torch.randn(2, 4, 25, 32)
    q = torch.randn(2, 32)
    model = Interaction()
    output = model(x, q)
    print(output.shape)