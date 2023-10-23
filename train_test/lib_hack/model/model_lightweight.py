import torch
from torch import nn
import math
import numpy as np


class TemporalGrounding(nn.Module):
    """Grounding module which can take in the visual and word features and outputs the probabilities of each visual
    feature belonging to that caption"""

    def __init__(self, visual_projection_switch, visual_feature_orig_dim, visual_feature_input_dim, feature_embed_dim,
                 linear_hidden_dim, num_heads, cap_input_dim, cap_embed_dim, device,
                 visual_dropout=0.3, cap_dropout=0.5):
        super(TemporalGrounding, self).__init__()
        torch.set_default_dtype(torch.float32)
        self.device = device
        self.visual_projection_switch = visual_projection_switch
        # Define layers for visual feature embedding
        self.feature_projection = nn.Sequential(
            nn.Linear(visual_feature_orig_dim, visual_feature_input_dim),
            nn.Dropout(visual_dropout),
            nn.ReLU()
        )
        self.visual_projection.apply(self.initialise_layer)
        self.self_att1 = MultiheadAttention(visual_feature_input_dim, feature_embed_dim, num_heads)
        self.self_att1.apply(self.initialise_layer)
        self.self_att2 = MultiheadAttention(visual_feature_input_dim, feature_embed_dim, num_heads)
        self.self_att2.apply(self.initialise_layer)
        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(visual_feature_input_dim, linear_hidden_dim),
            nn.Dropout(visual_dropout),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, visual_feature_input_dim)
        )
        self.linear_net.apply(self.initialise_layer)

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(visual_feature_input_dim)
        self.norm2 = nn.LayerNorm(visual_feature_input_dim)
        self.norm3 = nn.LayerNorm(visual_feature_input_dim)
        self.norm4 = nn.LayerNorm(visual_feature_input_dim)
        self.norm5 = nn.LayerNorm(visual_feature_input_dim)
        self.norm6 = nn.LayerNorm(visual_feature_input_dim)
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
            nn.Linear(linear_hidden_dim, visual_feature_input_dim),
        )
        self.attenuated_linear_net.apply(self.initialise_layer)

        self.linear_net2 = nn.Sequential(
            nn.Linear(visual_feature_input_dim, linear_hidden_dim),
            nn.Dropout(visual_dropout),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, visual_feature_input_dim)
        )
        self.linear_net2.apply(self.initialise_layer)

        # final linear layer for prediction
        self.pred = nn.Sequential(
            nn.Linear(visual_feature_input_dim, 1),
            nn.Sigmoid()
        )
        self.pred.apply(self.initialise_layer)
        self.zero_tensor = ZeroTensor()

    def visual_feature_embed(self, visual_features_in, mask=None):
        """Perform self attention over the input visual features in order to generate contextualise
        visual features"""
        att1 = self.self_att1(visual_features_in, mask=mask)
        resid = visual_features_in + self.dropout(att1)  # residual connection
        att1_out = self.norm1(resid)

        # MLP part
        lin1 = self.linear_net(att1_out)
        resid_lin1 = att1_out + self.dropout(lin1)
        lin1_out = self.norm2(resid_lin1)

        att2 = self.self_att2(lin1_out, mask=mask)
        resid_att2 = lin1_out + self.dropout(att2)
        att2_out = self.norm3(resid_att2)

        lin2 = self.linear_net2(att2_out)
        resid_lin2 = att2_out + self.dropout(lin2)
        lin2_out = self.norm4(resid_lin2)

        lin2_out = visual_features_in + lin2_out

        return lin2_out

    def caption_embed(self, bert_features, num_tokens):
        """Generate word embeddings in same space as the visual embeddings. Then average over them to
        get caption embeddings
        Input is bert features from dataloader, size (batch_size, 20, 768)"""
        word_embeds = self.cap_embed(bert_features)

        # generate 0 1 tensor to multiply by original
        zero_one = self.zero_tensor.forward(word_embeds.size()).to(self.device)

        for caption in range(word_embeds.size()[0]):
            zero_one[caption, 1: num_tokens[caption] - 1, :] = torch.ones((1, num_tokens[caption] - 2,
                                                                           word_embeds.size()[2]))

        embeddings_zeroed = torch.mul(word_embeds, zero_one)

        # Average over the word embeddings to produce a caption embedding
        summed_embeddings = torch.sum(embeddings_zeroed, 1)
        avg_divisor = (num_tokens - 2).unsqueeze(1)
        caption_embed = summed_embeddings / avg_divisor

        return caption_embed

    def text_attenuation(self, visual_features, caption_features):
        """visual features should have shape (1, 20, 2048)
        caption features should have shape (1, 2048)
        """
        # unsqueeze cap embeddings so that they will be subtracted from each
        caption_features = torch.unsqueeze(caption_features, dim=1)
        eps = 1e-10

        visual_features_norm = torch.div(visual_features,
                                         ((torch.linalg.norm(visual_features, dim=2, ord=2) + eps).unsqueeze(2)))
        caption_features_norm = torch.div(caption_features,
                                          ((torch.linalg.norm(caption_features, dim=2, ord=2) + eps).unsqueeze(2)))

        # hadamard product
        attenuated_feats = torch.mul(visual_features_norm, caption_features_norm)
        attenuated_feats = self.norm6(attenuated_feats)

        return attenuated_feats

    def post_attenuation_embedding(self, attenuated_features):
        """Further self attention over the caption attenuated visual features"""
        lin_out = self.attenuated_linear_net(attenuated_features)
        resid_lin = attenuated_features + self.dropout(lin_out)
        lin_out = self.norm5(resid_lin)

        return lin_out

    def final_prediction(self, features):
        pred = self.pred(features)
        return pred

    def single_video_embed(self, visual_input):
        if self.visual_projection_switch:
            visual_input = self.visual_projection(visual_input)
        visual_features = self.visual_feature_embed(visual_input)

        return visual_features

    def predict(self, visual_input, caption_input, num_tokens, val, attenuation=True):
        """Produce the final prediction on each visual feature"""
        if self.visual_projection_switch:
            visual_input = self.visual_projection(visual_input)
        visual_features = self.visual_feature_embed(visual_input, val)

        if attenuation:
            cap_features = self.caption_embed(caption_input, num_tokens)
            attenuated_features = self.text_attenuation(visual_features, cap_features)

            # WITH PREDICTION LAYER
            final_features = self.post_attenuation_embedding(attenuated_features)
        else:
            # WITH PREDICTION LAYER
            cap_features = self.caption_embed(caption_input, num_tokens)

            eps = 1e-10
            caption_features = torch.unsqueeze(cap_features, dim=1)
            cap_features_norm = torch.div(caption_features,
                                          ((torch.linalg.norm(caption_features, dim=2, ord=2) + eps).unsqueeze(2)))
            visual_features_norm = torch.div(visual_features,
                                             ((torch.linalg.norm(visual_features, dim=2, ord=2) + eps).unsqueeze(2)))
            cap_features_avg = torch.median(cap_features_norm, dim=2)[0].unsqueeze(2)
            visual_features_norm_scaled = torch.mul(visual_features_norm, cap_features_avg)
            visual_features_norm_scaled = self.norm6(visual_features_norm_scaled)

            final_features = self.post_attenuation_embedding(visual_features_norm_scaled)

        preds = self.final_prediction(final_features)

        return preds

    # Initialise the model weights
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, 0.1)
            # nn.init.constant_(layer.bias, 0)
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
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
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
