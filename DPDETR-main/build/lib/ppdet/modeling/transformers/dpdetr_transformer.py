# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Modified from detrex (https://github.com/IDEA-Research/detrex)
# Copyright 2022 The IDEA Authors. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention
from ..heads.detr_head import MLP
from .deformable_transformer import MSDeformableAttention_Position_Decoupled
from ..initializer import (linear_init_, constant_, xavier_uniform_, normal_, vector_,
                           bias_init_with_prob)
from .utils import (_get_clones,
                    get_decoupled_position_contrastive_denoising_training_group_obb, inverse_sigmoid)
import numpy as np

__all__ = ['DPDETR_obb_Transformer']


class PPMSDeformableAttention_obb_Position_Decoupled(MSDeformableAttention_Position_Decoupled):
    def forward(self,
                gt_meta,
                query,
                query_vis_sampling,
                query_ir_sampling,
                reference_points_vis,
                reference_points_ir,
                angle_vis,
                angle_ir,
                value,
                value_spatial_shapes,
                value_level_start_index,
                angle_max,
                half_pi_bin,
                value_mask=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None,
                mask_vis = None,
                flag = 'cls'):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets_vis = self.sampling_offsets_vis(query_vis_sampling).reshape(
            [bs, Len_q, self.num_heads, self.num_levels // 2, self.num_points, 2])
        sampling_offsets_ir = self.sampling_offsets_ir(query_ir_sampling).reshape(
            [bs, Len_q, self.num_heads, self.num_levels // 2, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        ## rotate
        rotate_part1_vis = paddle.concat([paddle.cos(angle_vis), paddle.sin(angle_vis)], axis=-1)
        rotate_part2_vis = paddle.concat([-paddle.sin(angle_vis), paddle.cos(angle_vis)], axis=-1)
        rotate_matrix_vis = paddle.stack([rotate_part1_vis, rotate_part2_vis], axis=-2)

        rotate_matrix_vis = paddle.broadcast_to(rotate_matrix_vis[:, :, None, None],
                                            [bs, Len_q, self.num_heads, self.num_levels//2, 2, 2])

        sampling_locations_vis = reference_points_vis[:, :, None, :, None, :2] + paddle.matmul(
            sampling_offsets_vis / self.num_points * reference_points_vis[:, :, None, :, None, 2:] * 0.5,
            rotate_matrix_vis)

        rotate_part1_ir = paddle.concat([paddle.cos(angle_ir), paddle.sin(angle_ir)], axis=-1)
        rotate_part2_ir = paddle.concat([-paddle.sin(angle_ir), paddle.cos(angle_ir)], axis=-1)
        rotate_matrix_ir = paddle.stack([rotate_part1_ir, rotate_part2_ir], axis=-2)

        rotate_matrix_ir = paddle.broadcast_to(rotate_matrix_ir[:, :, None, None],
                                                [bs, Len_q, self.num_heads, self.num_levels // 2, 2, 2])

        sampling_locations_ir = reference_points_ir[:, :, None, :, None, :2] + paddle.matmul(
            sampling_offsets_ir / self.num_points * reference_points_ir[:, :, None, :, None, 2:] * 0.5,
            rotate_matrix_ir)

        sampling_locations = paddle.concat([sampling_locations_vis, sampling_locations_ir], axis=3)

        if not isinstance(query, paddle.Tensor):
            from ppdet.modeling.transformers.utils import deformable_attention_core_func
            output = deformable_attention_core_func(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        else:

            value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
            value_level_start_index = paddle.to_tensor(value_level_start_index)
            output = self.ms_deformable_attn_core(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output



class TransformerDecoderLayer_obb_Decoupled(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 angle_max=None,
                 angle_proj=None,
                 weight_attr=None,
                 bias_attr=None,):
        super(TransformerDecoderLayer_obb_Decoupled, self).__init__()

        self.angle_max = angle_max
        self.angle_proj = angle_proj
        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention

        self.cross_attn_cls = PPMSDeformableAttention_obb_Position_Decoupled(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.cross_attn_visp = PPMSDeformableAttention_obb_Position_Decoupled(d_model, n_head, n_levels,
                                                                            n_points, 1.0)
        self.cross_attn_irp = PPMSDeformableAttention_obb_Position_Decoupled(d_model, n_head, n_levels,
                                                                            n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2_cls = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.norm2_visp = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.norm2_irp = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))


        # ffn
        self.linear1_cls = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.linear1_visp = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.linear1_irp = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2_cls = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.linear2_visp = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.linear2_irp = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3_cls = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.norm3_visp = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.norm3_irp = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1_cls)
        linear_init_(self.linear1_visp)
        linear_init_(self.linear1_irp)
        linear_init_(self.linear2_cls)
        linear_init_(self.linear2_visp)
        linear_init_(self.linear2_irp)
        xavier_uniform_(self.linear1_cls.weight)
        xavier_uniform_(self.linear1_visp.weight)
        xavier_uniform_(self.linear1_irp.weight)
        xavier_uniform_(self.linear2_cls.weight)
        xavier_uniform_(self.linear2_visp.weight)
        xavier_uniform_(self.linear2_irp.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn_cls(self, tgt):
        return self.linear2_cls(self.dropout3(self.activation(self.linear1_cls(tgt))))
    def forward_ffn_visp(self, tgt):
        return self.linear2_visp(self.dropout3(self.activation(self.linear1_visp(tgt))))
    def forward_ffn_irp(self, tgt):
        return self.linear2_irp(self.dropout3(self.activation(self.linear1_irp(tgt))))

    def forward(self,
                gt_meta,
                tgt_cls,
                tgt_visp,
                tgt_irp,
                reference_points_vis,
                reference_points_ir,
                angle_vis,
                angle_ir,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                angle_max,
                half_pi_bin,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed_vis=None,
                query_pos_embed_ir=None,
                topk_ind_mask = None,
                topk_score = None,
                mask_vis = None,
                dn_meta = None):
        # self attention
        query_pos_embed = query_pos_embed_vis + query_pos_embed_ir
        bs, len_q, len_c = tgt_cls.shape
        if self.training and dn_meta:
            tgt_cls_dn = tgt_cls[:,:dn_meta['dn_num_split'][0],:]
            tgt_cls_query = tgt_cls[:,dn_meta['dn_num_split'][0]:,:]
            tgt_visp_dn = tgt_visp[:, :dn_meta['dn_num_split'][0], :]
            tgt_visp_query = tgt_visp[:, dn_meta['dn_num_split'][0]:, :]
            tgt_irp_dn = tgt_irp[:, :dn_meta['dn_num_split'][0], :]
            tgt_irp_query = tgt_irp[:, dn_meta['dn_num_split'][0]:, :]

            # q, k
            q_cls_dn = paddle.reshape(tgt_cls_dn,[bs, dn_meta['dn_num_group'],
                                        dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'], len_c])
            q_visp_dn = paddle.reshape(self.with_pos_embed(tgt_visp_dn, query_pos_embed_vis[:,:dn_meta['dn_num_split'][0],:]),
                                       [bs, dn_meta['dn_num_group'],
                                        dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'], len_c]
                                       )
            q_irp_dn = paddle.reshape(self.with_pos_embed(tgt_irp_dn, query_pos_embed_ir[:,:dn_meta['dn_num_split'][0],:]),
                                      [bs, dn_meta['dn_num_group'],
                                       dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'], len_c]
                                      )

            q_cls_query = tgt_cls_query
            q_visp_query = self.with_pos_embed(tgt_visp_query, query_pos_embed_vis[:,dn_meta['dn_num_split'][0]:,:])
            q_irp_query = self.with_pos_embed(tgt_irp_query, query_pos_embed_ir[:,dn_meta['dn_num_split'][0]:,:])


            tgt_cls_dn = paddle.reshape(tgt_cls_dn, [bs, dn_meta['dn_num_group'], dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'], len_c])
            tgt_visp_dn = paddle.reshape(tgt_visp_dn, [bs, dn_meta['dn_num_group'],
                                                     dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'], len_c])
            tgt_irp_dn = paddle.reshape(tgt_irp_dn, [bs, dn_meta['dn_num_group'],
                                                     dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'], len_c])


            #new dn
            tgt_dn = paddle.empty(shape=(bs,1,dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'],len_c))
            q_dn = paddle.empty(shape=(bs,1,dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'],len_c))
            #k_dn = paddle.empty(shape=())
            for i in range(dn_meta['dn_num_group']):
                tgt_dn = paddle.concat([tgt_dn, tgt_cls_dn[:,i:i+1,:,:]], axis=1)
                tgt_dn = paddle.concat([tgt_dn, tgt_visp_dn[:,i:i+1,:,:]], axis=1)
                tgt_dn = paddle.concat([tgt_dn, tgt_irp_dn[:,i:i+1,:,:]], axis=1)

                q_dn = paddle.concat([q_dn, q_cls_dn[:,i:i+1,:,:]], axis=1)
                q_dn = paddle.concat([q_dn, q_visp_dn[:,i:i+1,:,:]], axis=1)
                q_dn = paddle.concat([q_dn, q_irp_dn[:,i:i+1,:,:]], axis=1)

            tgt_dn = tgt_dn[:, 1:,:,:]
            q_dn = q_dn[:, 1:, :, :]

            tgt_dn = paddle.reshape(tgt_dn, [bs,dn_meta['dn_num_split'][0] * 3, len_c ])
            q_dn = paddle.reshape(q_dn, [bs, dn_meta['dn_num_split'][0] * 3, len_c])
            # k_dn = paddle.reshape(k_dn, [bs, dn_meta['dn_num_split'][0] * 3, len_c])
            tgt_query = paddle.concat([tgt_cls_query, tgt_visp_query, tgt_irp_query], axis=1)
            q_query = paddle.concat([q_cls_query, q_visp_query, q_irp_query], axis=1)
            # k_query = paddle.concat([k_cls_query, k_visp_query, k_irp_query], axis=1)

            q = k = paddle.concat([q_dn, q_query], axis=1)
            tgt = paddle.concat([tgt_dn, tgt_query], axis=1)

        else:
            q_cls = k_cls = tgt_cls
            q_visp = k_visp = self.with_pos_embed(tgt_visp, query_pos_embed_vis)
            q_irp = k_irp = self.with_pos_embed(tgt_irp, query_pos_embed_ir)
            q = paddle.concat([q_cls, q_visp, q_irp], axis=1)
            k = paddle.concat([k_cls, k_visp, k_irp], axis=1)
            tgt = paddle.concat([tgt_cls, tgt_visp, tgt_irp], axis=1)


        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)


        if self.training and dn_meta:
            tgt_dn = tgt[:,:dn_meta['dn_num_split'][0]*3,:]
            tgt_query = tgt[:,dn_meta['dn_num_split'][0]*3:,:]

            # split query
            tgt_cls_query, tgt_visp_query, tgt_irp_query = paddle.split(tgt_query, num_or_sections=3, axis=1)

            # split dn
            tgt_dn = paddle.reshape(tgt_dn, [bs,dn_meta['dn_num_group'] * 3, dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'] , len_c ])
            tgt_cls_dn = paddle.empty(shape=(bs,1,dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'],len_c))
            tgt_visp_dn = paddle.empty(shape=(bs,1,dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'],len_c))
            tgt_irp_dn = paddle.empty(shape=(bs,1,dn_meta['dn_num_split'][0] // dn_meta['dn_num_group'],len_c))
            for i in range(0, dn_meta['dn_num_group'] * 3, 3):
                tgt_cls_dn = paddle.concat([tgt_cls_dn, tgt_dn[:,i:i+1,:,:]], axis=1)
                tgt_visp_dn = paddle.concat([tgt_visp_dn, tgt_dn[:,i+1:i+2,:,:]], axis=1)
                tgt_irp_dn = paddle.concat([tgt_irp_dn, tgt_dn[:, i+2:i + 3, :, :]], axis=1)
            tgt_cls_dn = tgt_cls_dn[:,1:,:,:]
            tgt_visp_dn = tgt_visp_dn[:,1:,:,:]
            tgt_irp_dn = tgt_irp_dn[:,1:,:,:]
            tgt_cls_dn = paddle.reshape(tgt_cls_dn, [bs, dn_meta['dn_num_split'][0], len_c])
            tgt_visp_dn = paddle.reshape(tgt_visp_dn, [bs, dn_meta['dn_num_split'][0], len_c])
            tgt_irp_dn = paddle.reshape(tgt_irp_dn, [bs, dn_meta['dn_num_split'][0], len_c])

            tgt_cls = paddle.concat([tgt_cls_dn, tgt_cls_query], axis=1)
            tgt_visp = paddle.concat([tgt_visp_dn, tgt_visp_query], axis=1)
            tgt_irp = paddle.concat([tgt_irp_dn, tgt_irp_query], axis=1)

        else:
            #split
            tgt_cls, tgt_visp, tgt_irp = paddle.split(tgt, num_or_sections=3, axis=1)

        # cross attention
        tgt2_cls = self.cross_attn_cls(gt_meta,
            self.with_pos_embed(tgt_cls, query_pos_embed),self.with_pos_embed(tgt_cls, query_pos_embed_vis),
                                       self.with_pos_embed(tgt_cls, query_pos_embed_ir),
                                       reference_points_vis,reference_points_ir, angle_vis,angle_ir, memory,
            memory_spatial_shapes, memory_level_start_index,angle_max,half_pi_bin, memory_mask,topk_ind_mask,topk_score, mask_vis=mask_vis, flag='cls')

        tgt2_visp = self.cross_attn_visp(gt_meta,
                                       self.with_pos_embed(tgt_visp, query_pos_embed_vis),
                                       self.with_pos_embed(tgt_visp, query_pos_embed_vis),
                                       self.with_pos_embed(tgt_visp, query_pos_embed_ir),
                                       reference_points_vis, reference_points_ir, angle_vis, angle_ir, memory,
                                       memory_spatial_shapes, memory_level_start_index, angle_max, half_pi_bin,
                                       memory_mask, topk_ind_mask, topk_score, mask_vis=mask_vis, flag='visp')
        tgt2_irp = self.cross_attn_irp(gt_meta,
                                         self.with_pos_embed(tgt_irp, query_pos_embed_ir),
                                         self.with_pos_embed(tgt_irp, query_pos_embed_vis),
                                         self.with_pos_embed(tgt_irp, query_pos_embed_ir),
                                         reference_points_vis, reference_points_ir, angle_vis, angle_ir, memory,
                                         memory_spatial_shapes, memory_level_start_index, angle_max, half_pi_bin,
                                         memory_mask, topk_ind_mask, topk_score, mask_vis=mask_vis, flag='irp')

        tgt_cls = tgt_cls + self.dropout2(tgt2_cls)
        tgt_cls = self.norm2_cls(tgt_cls)
        tgt_visp = tgt_visp + self.dropout2(tgt2_visp)
        tgt_visp = self.norm2_visp(tgt_visp)
        tgt_irp = tgt_irp + self.dropout2(tgt2_irp)
        tgt_irp = self.norm2_irp(tgt_irp)

        # ffn
        tgt2_cls = self.forward_ffn_cls(tgt_cls)
        tgt_cls = tgt_cls + self.dropout4(tgt2_cls)
        tgt_cls = self.norm3_cls(tgt_cls)

        tgt2_visp = self.forward_ffn_visp(tgt_visp)
        tgt_visp = tgt_visp + self.dropout4(tgt2_visp)
        tgt_visp = self.norm3_visp(tgt_visp)

        tgt2_irp = self.forward_ffn_irp(tgt_irp)
        tgt_irp = tgt_irp + self.dropout4(tgt2_irp)
        tgt_irp = self.norm3_irp(tgt_irp)

        return tgt_cls, tgt_visp, tgt_irp


class TransformerDecoder_obb_Decoupled(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, angle_max, angle_proj, half_pi_bin,eval_idx=-1):
        super(TransformerDecoder_obb_Decoupled, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.angle_max = angle_max
        self.angle_proj = angle_proj
        self.half_pi_bin = half_pi_bin

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact_vis,
                ref_points_unact_ir,
                ref_angle_cls_vis,
                ref_angle_cls_ir,
                ref_angle_vis,
                ref_angle_ir,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head_vis,
                bbox_head_ir,
                score_head,
                angle_head_vis,
                angle_head_ir,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None,
                mask_vis = None,
                dn_meta = None,
                learnt_init_query = None):

        if learnt_init_query:
            output_cls = tgt[0]
            output_visp = tgt[1]
            output_irp = tgt[2]
        else:
            output_cls = tgt.clone()
            output_visp = tgt.clone()
            output_irp = tgt.clone()
        dec_out_bboxes_vis = []
        dec_out_bboxes_ir = []
        dec_out_logits = []
        dec_out_angles_cls_vis = []
        dec_out_angles_cls_ir = []
        dec_out_angles_vis = []
        dec_out_angles_ir = []
        ref_points_detach_vis = F.sigmoid(ref_points_unact_vis)
        ref_points_detach_ir = F.sigmoid(ref_points_unact_ir)

        # #get angle
        b, l = ref_angle_vis.shape[:2]
        angle_vis = ref_angle_vis
        angle_ir = ref_angle_ir

        gt_meta['layer'] = 0

        for i, layer in enumerate(self.layers):

            ref_points_input_vis = ref_points_detach_vis.unsqueeze(2)
            ref_points_input_ir = ref_points_detach_ir.unsqueeze(2)

            query_pos_embed_vis = query_pos_head(ref_points_detach_vis)
            query_pos_embed_ir = query_pos_head(ref_points_detach_ir)

            output_cls, output_visp, output_irp = layer(gt_meta, output_cls,output_visp,output_irp, ref_points_input_vis,ref_points_input_ir, angle_vis,angle_ir, memory,
                           memory_spatial_shapes, memory_level_start_index,self.angle_max, self.half_pi_bin,
                           attn_mask, memory_mask, query_pos_embed_vis,query_pos_embed_ir,topk_ind_mask,topk_score, mask_vis,dn_meta)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox_vis = F.sigmoid(bbox_head_vis[i](output_visp) + inverse_sigmoid(
                ref_points_detach_vis))
            angle_cls_vis = angle_head_vis[i](output_visp)
            angle_vis = F.softmax(angle_cls_vis.reshape([b, l, 1, self.angle_max + 1
                                              ])).matmul(self.angle_proj)

            inter_ref_bbox_ir = F.sigmoid(bbox_head_ir[i](output_irp) + inverse_sigmoid(
                ref_points_detach_ir))
            angle_cls_ir = angle_head_ir[i](output_irp)
            angle_ir = F.softmax(angle_cls_ir.reshape([b, l, 1, self.angle_max + 1
                                                         ])).matmul(self.angle_proj)

            if self.training:
                dec_out_logits.append(score_head[i](output_cls))
                dec_out_angles_cls_vis.append(angle_cls_vis)
                dec_out_angles_vis.append(angle_vis)
                dec_out_angles_cls_ir.append(angle_cls_ir)
                dec_out_angles_ir.append(angle_ir)
                if i == 0:
                    dec_out_bboxes_vis.append(inter_ref_bbox_vis)
                    dec_out_bboxes_ir.append(inter_ref_bbox_ir)
                else:
                    dec_out_bboxes_vis.append(
                        F.sigmoid(bbox_head_vis[i](output_visp) + inverse_sigmoid(
                            ref_points_vis)))
                    dec_out_bboxes_ir.append(
                        F.sigmoid(bbox_head_ir[i](output_irp) + inverse_sigmoid(
                            ref_points_ir)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output_cls))
                dec_out_bboxes_vis.append(inter_ref_bbox_vis)
                dec_out_angles_cls_vis.append(angle_cls_vis)
                dec_out_angles_vis.append(angle_vis)
                dec_out_bboxes_ir.append(inter_ref_bbox_ir)
                dec_out_angles_cls_ir.append(angle_cls_ir)
                dec_out_angles_ir.append(angle_ir)
                break

            ref_points_vis = inter_ref_bbox_vis
            ref_points_ir = inter_ref_bbox_ir
            ref_points_detach_vis = inter_ref_bbox_vis.detach(
            ) if self.training else inter_ref_bbox_vis
            ref_points_detach_ir = inter_ref_bbox_ir.detach(
            ) if self.training else inter_ref_bbox_ir


        return paddle.stack(dec_out_bboxes_vis),paddle.stack(dec_out_bboxes_ir), paddle.stack(dec_out_logits), paddle.stack(dec_out_angles_cls_vis),paddle.stack(dec_out_angles_cls_ir), paddle.stack(dec_out_angles_vis),paddle.stack(dec_out_angles_ir)


@register
class DPDETR_obb_Transformer(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 angle_noise_scale=0.03,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 key_aware = False,
                 proj_all = False,
                 split_attention = False):
        super(DPDETR_obb_Transformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.proj_all = proj_all

        self.half_pi = paddle.to_tensor(
            [1.5707963267948966], dtype=paddle.float32)
        self.angle_max = 90
        self.half_pi_bin = self.half_pi / self.angle_max
        angle_proj = paddle.linspace(0, self.angle_max, self.angle_max + 1)
        self.angle_proj = angle_proj * self.half_pi_bin


        if self.proj_all:
            # backbone feature projection
            self._build_input_proj_layer(backbone_visir_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_obb_Decoupled(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points,self.angle_max, self.angle_proj)
        self.decoder = TransformerDecoder_obb_Decoupled(hidden_dim, decoder_layer,
                                          num_decoder_layers,self.angle_max, self.angle_proj ,self.half_pi_bin, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.angle_noise_scale = angle_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed_cls = nn.Embedding(num_queries, hidden_dim)
            self.tgt_embed_visp = nn.Embedding(num_queries, hidden_dim)
            self.tgt_embed_irp = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_vis = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.enc_bbox_head_ir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # angle head
        self.enc_angle_head_vis = nn.Linear(hidden_dim, self.angle_max + 1)
        self.enc_angle_head_ir = nn.Linear(hidden_dim, self.angle_max + 1)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head_vis = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head_ir = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # angle head
        self.dec_angle_head_vis = nn.LayerList([
            nn.Linear(hidden_dim, self.angle_max + 1)
            for _ in range(num_decoder_layers)
        ])
        self.dec_angle_head_ir = nn.LayerList([
            nn.Linear(hidden_dim, self.angle_max + 1)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        bias_angle = [10.] + [1.] * self.angle_max
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_angle_head_vis.weight)
        constant_(self.enc_angle_head_ir.weight)
        vector_(self.enc_angle_head_vis.bias, bias_angle)
        vector_(self.enc_angle_head_ir.bias, bias_angle)
        constant_(self.enc_bbox_head_vis.layers[-1].weight)
        constant_(self.enc_bbox_head_vis.layers[-1].bias)
        constant_(self.enc_bbox_head_ir.layers[-1].weight)
        constant_(self.enc_bbox_head_ir.layers[-1].bias)
        for cls_, reg_vis, reg_ir, angle_vis, angle_ir in zip(self.dec_score_head, self.dec_bbox_head_vis,self.dec_bbox_head_ir, self.dec_angle_head_vis,self.dec_angle_head_ir):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_vis.layers[-1].weight)
            constant_(reg_vis.layers[-1].bias)
            constant_(angle_vis.weight)
            vector_(angle_vis.bias, bias_angle)
            constant_(reg_ir.layers[-1].weight)
            constant_(reg_ir.layers[-1].bias)
            constant_(angle_ir.weight)
            vector_(angle_ir.bias, bias_angle)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed_cls.weight)
            xavier_uniform_(self.tgt_embed_visp.weight)
            xavier_uniform_(self.tgt_embed_irp.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):

        visir_feats = vis_feats + ir_feats

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact_vis,denoising_bbox_unact_ir,denosing_angle_vis,denosing_angle_ir, attn_mask, dn_meta = \
                get_decoupled_position_contrastive_denoising_training_group_obb(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale,
                                            self.angle_noise_scale)
        else:
            denoising_class, denoising_bbox_unact_vis,denoising_bbox_unact_ir,denosing_angle_vis,denosing_angle_ir, attn_mask, dn_meta = None, None, None, None, None, None, None



        target, init_ref_points_unact_vis,init_ref_points_unact_ir,init_ref_angle_cls_vis,init_ref_angle_cls_ir, init_ref_angle_vis,init_ref_angle_ir, enc_topk_bboxes_vis,enc_topk_bboxes_ir, enc_topk_logits,enc_topk_angles_cls_vis,enc_topk_angles_cls_ir, topk_ind_mask, topk_score, mask_vis = \
            self._get_decoder_input(gt_meta,
                                    visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class,
                                    denoising_bbox_unact_vis,denoising_bbox_unact_ir,denosing_angle_vis,denosing_angle_ir)

        # decoder
        out_bboxes_vis,out_bboxes_ir, out_logits,out_angles_cls_vis,out_angles_cls_ir, out_angles_vis,out_angles_ir = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact_vis,
            init_ref_points_unact_ir,
            init_ref_angle_cls_vis,
            init_ref_angle_cls_ir,
            init_ref_angle_vis,
            init_ref_angle_ir,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head_vis,
            self.dec_bbox_head_ir,
            self.dec_score_head,
            self.dec_angle_head_vis,
            self.dec_angle_head_ir,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score,
            mask_vis = mask_vis,
            dn_meta = dn_meta,
            learnt_init_query = self.learnt_init_query)
        return (out_bboxes_vis,out_bboxes_ir, out_logits, out_angles_cls_vis,out_angles_cls_ir, out_angles_vis,out_angles_ir, enc_topk_bboxes_vis,enc_topk_bboxes_ir, enc_topk_logits, enc_topk_angles_cls_vis,enc_topk_angles_cls_ir,self.angle_max,self.angle_proj,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact_vis=None,
                           denoising_bbox_unact_ir=None,
                           denosing_angle_vis=None,
                           denosing_angle_ir=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact_vis = self.enc_bbox_head_vis(output_memory) + anchors
        enc_outputs_coord_unact_ir = self.enc_bbox_head_ir(output_memory) + anchors
        # pred angle
        enc_outputs_angle_cls_vis = self.enc_angle_head_vis(output_memory)
        enc_outputs_angle_cls_ir = self.enc_angle_head_ir(output_memory)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)

        ## record
        topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        topk_ind_mask.stop_gradient = True
        mask_vis = None


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact_vis = paddle.gather_nd(enc_outputs_coord_unact_vis,
                                                  topk_ind)  # unsigmoided.
        reference_points_unact_ir = paddle.gather_nd(enc_outputs_coord_unact_ir,
                                                      topk_ind)  # unsigmoided.
        enc_topk_bboxes_vis = F.sigmoid(reference_points_unact_vis)
        enc_topk_bboxes_ir = F.sigmoid(reference_points_unact_ir)

        # angle
        enc_topk_angles_vis = paddle.gather_nd(enc_outputs_angle_cls_vis, topk_ind)
        enc_topk_angles_ir = paddle.gather_nd(enc_outputs_angle_cls_ir, topk_ind)

        # get angle
        b, l = enc_topk_angles_vis.shape[:2]
        reference_angle_vis = F.softmax(enc_topk_angles_vis.reshape([b, l, 1, self.angle_max + 1
                                                             ])).matmul(self.angle_proj)
        reference_angle_ir = F.softmax(enc_topk_angles_ir.reshape([b, l, 1, self.angle_max + 1
                                                                    ])).matmul(self.angle_proj)

        if denoising_bbox_unact_vis is not None:
            reference_points_unact_vis = paddle.concat(
                [denoising_bbox_unact_vis, reference_points_unact_vis], 1)
            reference_points_unact_ir = paddle.concat(
                [denoising_bbox_unact_ir, reference_points_unact_ir], 1)
            reference_angle_vis = paddle.concat(
                [denosing_angle_vis, reference_angle_vis], 1
            )
            reference_angle_ir = paddle.concat(
                [denosing_angle_ir, reference_angle_ir], 1
            )
        if self.training:
            reference_points_unact_vis = reference_points_unact_vis.detach()
            reference_angle_cls_vis = enc_topk_angles_vis.detach()
            reference_angle_vis = reference_angle_vis.detach()
            reference_points_unact_ir = reference_points_unact_ir.detach()
            reference_angle_cls_ir = enc_topk_angles_ir.detach()
            reference_angle_ir = reference_angle_ir.detach()
        else:
            reference_angle_cls_vis = enc_topk_angles_vis
            reference_angle_cls_ir = enc_topk_angles_ir
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target_cls = self.tgt_embed_cls.weight.unsqueeze(0).tile([bs, 1, 1])
            target_visp = self.tgt_embed_visp.weight.unsqueeze(0).tile([bs, 1, 1])
            target_irp = self.tgt_embed_irp.weight.unsqueeze(0).tile([bs, 1, 1])
            target = [target_cls, target_visp, target_irp]
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            if self.learnt_init_query:
                target_cls = paddle.concat([denoising_class, target_cls], 1)
                target_visp = paddle.concat([denoising_class, target_visp], 1)
                target_irp = paddle.concat([denoising_class, target_irp], 1)
                target = [target_cls, target_visp, target_irp]
            else:
                target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact_vis,reference_points_unact_ir,reference_angle_cls_vis,reference_angle_cls_ir, reference_angle_vis,reference_angle_ir, enc_topk_bboxes_vis,enc_topk_bboxes_ir, enc_topk_logits, enc_topk_angles_vis,enc_topk_angles_ir, topk_ind_mask, topk_score, mask_vis


