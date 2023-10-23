import torch
from torch import nn
import math
import numpy as np


class TemporalGroundingCross(nn.Module):
    """Grounding module which can take in the visual and word features and outputs the probabilities of each visual
    feature belonging to that caption"""
    def __init__(self, visual_projection_switch, visual_feature_orig_dim, visual_feature_input_dim, feature_embed_dim,
                 linear_hidden_dim, num_heads, cap_input_dim, cap_embed_dim, device,
                 visual_dropout=0.3, cap_dropout=0.5):
        super(TemporalGroundingCross, self).__init__()
        torch.set_default_dtype(torch.float32)
        self.device = device
        self.num_heads = num_heads
        self.visual_projection_switch = visual_projection_switch
        # Define layers for visual embedding
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_feature_orig_dim, visual_feature_input_dim),
            nn.Dropout(visual_dropout),
            nn.ReLU()
        )
        self.visual_projection.apply(self.initialise_layer)
        self.self_att1 = MultiheadAttention(visual_feature_input_dim, feature_embed_dim, num_heads)
        self.self_att1.apply(self.initialise_layer)
        self.self_att2 = MultiheadAttention(visual_feature_input_dim, feature_embed_dim, num_heads)
        self.self_att2.apply(self.initialise_layer)
        self.cross_att1 = MultiheadAttention(visual_feature_input_dim, feature_embed_dim, num_heads)
        self.cross_att1.apply(self.initialise_layer)
        self.cross_att2 = MultiheadAttention(visual_feature_input_dim, feature_embed_dim, num_heads)
        self.cross_att2.apply(self.initialise_layer)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(visual_feature_input_dim, linear_hidden_dim),
            nn.Dropout(visual_dropout),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, visual_feature_input_dim)
        )
        self.linear_net.apply(self.initialise_layer)

        # Layers to apply in between the main layers#
        self.norm1 = nn.LayerNorm(visual_feature_input_dim)
        self.norm2 = nn.LayerNorm(visual_feature_input_dim)
        self.norm3 = nn.LayerNorm(visual_feature_input_dim)
        self.norm4 = nn.LayerNorm(visual_feature_input_dim)
        self.norm5 = nn.LayerNorm(visual_feature_input_dim)
        self.norm6 = nn.LayerNorm(visual_feature_input_dim)
        self.norm7 = nn.LayerNorm(visual_feature_input_dim)
        self.norm8 = nn.LayerNorm(visual_feature_input_dim)
        self.dropout = nn.Dropout(visual_dropout)

        # Define layers for caption embedding
        self.cap_embed = nn.Sequential(
            nn.Linear(cap_input_dim, cap_embed_dim),
            nn.Dropout(cap_dropout),
            nn.ReLU()
        )
        self.cap_embed.apply(self.initialise_layer)

        # Define layers for post attenuation embedding
        self.attenuated_linear_net = nn.Sequential(
            nn.Linear(visual_feature_input_dim, linear_hidden_dim),
            nn.Dropout(visual_dropout),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, visual_feature_input_dim)
        )
        self.attenuated_linear_net.apply(self.initialise_layer)

        self.linear_net2 = nn.Sequential(
            nn.Linear(visual_feature_input_dim, linear_hidden_dim),
            nn.Dropout(visual_dropout),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, visual_feature_input_dim)
        )
        self.linear_net2.apply(self.initialise_layer)

        self.linear_net3 = nn.Sequential(
             nn.Linear(visual_feature_input_dim, linear_hidden_dim),
             nn.Dropout(visual_dropout),
             nn.ReLU(),
             nn.Linear(linear_hidden_dim, visual_feature_input_dim)
             # note how this returns something of same size as input
        )
        self.linear_net3.apply(self.initialise_layer)

        self.linear_net4 = nn.Sequential(
            nn.Linear(visual_feature_input_dim, linear_hidden_dim),
            nn.Dropout(visual_dropout),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, visual_feature_input_dim)
            # note how this returns something of same size as input
        )
        self.linear_net4.apply(self.initialise_layer)

        # final linear layer for prediction
        self.pred = nn.Sequential(
            nn.Linear(visual_feature_input_dim, 1),
            nn.Sigmoid()
        )
        self.pred.apply(self.initialise_layer)
        self.zero_tensor = ZeroTensor()

    def word_embeds(self, bert_features, num_tokens):
        """Generate word embeddings in same space as the visual embeddings.
           Input is bert features from dataloader, size (batch_size, 20, 768)"""
        word_embeds = self.cap_embed(bert_features)

        # generate 0 1 tensor to multiply by original
        zero_one = self.zero_tensor.forward(word_embeds.size()).to(self.device)

        for caption in range(word_embeds.size()[0]):
            zero_one[caption, 1: num_tokens[caption] - 1, :] = torch.ones((1, num_tokens[caption] - 2,
                                                                           word_embeds.size()[2]))

        embeddings_zeroed = torch.mul(word_embeds, zero_one)

        return embeddings_zeroed

    def visual_word_interaction(self, visual_features_in, word_embeds_in, attenuation):
        full_features = torch.cat((word_embeds_in, visual_features_in), dim=1)

        batch_size = visual_features_in.size(0)

        mask_self = torch.zeros((batch_size, self.num_heads, full_features.size(1),
                                 full_features.size(1))).to(self.device)
        mask_cross = torch.zeros((batch_size, self.num_heads, full_features.size(1),
                                  full_features.size(1))).to(self.device)

        word_word_mask = torch.ones(batch_size, self.num_heads, word_embeds_in.size(1), word_embeds_in.size(1))
        visual_visual_mask = torch.ones(batch_size, self.num_heads, visual_features_in.size(1),
                                        visual_features_in.size(1))

        mask_self[:, :, word_embeds_in.size(1):, word_embeds_in.size(1):] = visual_visual_mask
        mask_self[:, :, :word_embeds_in.size(1), :word_embeds_in.size(1)] = word_word_mask

        word_visual_mask = torch.ones(batch_size, self.num_heads, word_embeds_in.size(1), visual_features_in.size(1))
        visual_word_mask = torch.ones(batch_size, self.num_heads, visual_features_in.size(1), word_embeds_in.size(1))

        mask_cross[:, :, :word_embeds_in.size(1), word_embeds_in.size(1):] = word_visual_mask
        mask_cross[:, :, word_embeds_in.size(1):, :word_embeds_in.size(1)] = visual_word_mask

        mask_no_attenuation = torch.zeros((batch_size, self.num_heads, full_features.size(1),
                                           full_features.size(1))).to(self.device)

        # Self attention 1
        self_att1 = self.self_att1(full_features, mask=mask_self)
        resid = full_features + self.dropout(self_att1)  # residual connection
        self_att1_out = self.norm1(resid)

        # MLP part
        lin1 = self.linear_net(self_att1_out)
        resid_lin1 = self_att1_out + self.dropout(lin1)
        lin1_out = self.norm2(resid_lin1)

        # Cross attention 1
        mask_cross = torch.ones((batch_size, self.num_heads, lin1_out.size(1), lin1_out.size(1))).to(self.device)
        if attenuation:
            cross_att1 = self.cross_att1(lin1_out, mask=mask_cross)
        else:
            cross_att1 = self.cross_att1(lin1_out, mask=mask_no_attenuation)
        resid = lin1_out + self.dropout(cross_att1)  # residual connection
        cross_att1_out = self.norm3(resid)

        # MLP part
        lin2 = self.linear_net2(cross_att1_out)
        resid_lin2 = cross_att1_out + self.dropout(lin2)
        lin2_out = self.norm4(resid_lin2)

        # Self attention 2
        mask_self = torch.ones((batch_size, self.num_heads, lin2_out.size(1), lin2_out.size(1))).to(self.device)
        self_att2 = self.self_att2(lin2_out, mask=mask_self)
        resid = lin2_out + self.dropout(self_att2)  # residual connection
        self_att2_out = self.norm5(resid)

        # MLP part
        lin3 = self.linear_net3(self_att2_out)
        resid_lin3 = self_att2_out + self.dropout(lin3)
        lin3_out = self.norm6(resid_lin3)

        # Cross attention 2
        mask_cross = torch.ones((batch_size, self.num_heads, lin3_out.size(1), lin3_out.size(1))).to(self.device)
        if attenuation:
            cross_att2 = self.cross_att2(lin3_out, mask=mask_cross)
        else:
            cross_att2 = self.cross_att2(lin3_out, mask=mask_no_attenuation)
        resid = lin3_out + self.dropout(cross_att2)  # residual connection
        cross_att2_out = self.norm7(resid)

        # MLP part
        lin4 = self.linear_net4(cross_att2_out)
        resid_lin4 = cross_att2_out + self.dropout(lin4)
        lin4_out = self.norm8(resid_lin4)

        lin4_out = full_features + lin4_out

        return lin4_out

    def post_attenuation_embedding(self, attenuated_features):
        """Further self attention over the caption attenuated visual features"""
        # MLP part without attention
        lin_out = self.attenuated_linear_net(attenuated_features)
        resid_lin = attenuated_features + self.dropout(lin_out)
        lin_out = self.norm5(resid_lin)

        return lin_out

    def final_prediction(self, features):
        pred = self.pred(features)
        return pred

    def predict(self, visual_input, caption_input, num_tokens, val, attenuation=True):
        """Produce the final prediction on each visual feature"""
        if self.visual_projection_switch:
            visual_input = self.visual_projection(visual_input)

        if attenuation:
            word_embeds = self.word_embeds(caption_input, num_tokens)
            attenuated_features = self.visual_word_interaction(visual_input, word_embeds, attenuation)
            num_word_tokens = word_embeds.size(1)
            attenuated_features = attenuated_features[:, num_word_tokens:, :]
            final_features = self.post_attenuation_embedding(attenuated_features)
        else:
            final_features = self.post_attenuation_embedding(visual_input)

        preds = self.final_prediction(final_features)

        return preds

    # Initialise the model weights
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, 0.1)
        elif hasattr(layer, "weight"):
            nn.init.normal_(layer.weight, 0.0, 0.5)


class ZeroTensor(nn.Module):
    def __init__(self):
        """
        Generate zero tensor for multiplying with the embeddings to zero out padding, cls and sep
        """
        super().__init__()

    def forward(self, word_embeds_size):
        # generate 0 1 tensor to multiply by original
        zero_one = torch.zeros(word_embeds_size)
        return zero_one


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        softmax = nn.Softmax(dim=-1)
        attention = softmax(attn_logits)
        values = torch.matmul(attention, v)
        return values, attention
