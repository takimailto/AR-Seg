"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionAwareSVDSpatialReduction(nn.Module):
    def __init__(self, in_channels=256, spatial_dim=64, reduced_dim=256, 
                 svd_energy_ratio=0.95, use_learnable_pos=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.spatial_dim = spatial_dim
        self.reduced_dim = reduced_dim
        self.svd_energy_ratio = svd_energy_ratio
        
        if use_learnable_pos:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, in_channels, spatial_dim, spatial_dim) * 0.02
            )
        else:
            self.register_buffer('pos_encoding', 
                self._create_sinusoidal_pos_encoding(spatial_dim, spatial_dim, in_channels))
  
        self.importance_estimator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.svd_value_enhancer = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.spatial_fusion = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim // 2),
            nn.LayerNorm(reduced_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(reduced_dim // 2, reduced_dim),
            nn.LayerNorm(reduced_dim)
        )

        self.residual_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _create_sinusoidal_pos_encoding(self, H, W, C):
        pos_encoding = torch.zeros(1, C, H, W)
        
        y_coords = torch.arange(H).float().reshape(H, 1)
        x_coords = torch.arange(W).float().reshape(1, W)
        
        for c in range(C):
            if c % 4 == 0:
                freq = 1.0 / (10000 ** ((c % 64) / 64))
                pos_encoding[0, c] = torch.sin(y_coords * freq).repeat(1, W)
            elif c % 4 == 1:
                freq = 1.0 / (10000 ** ((c % 64) / 64))
                pos_encoding[0, c] = torch.cos(x_coords * freq).repeat(H, 1)
            elif c % 4 == 2:
                y_norm = (y_coords / (H-1)) * 2 - 1
                x_norm = (x_coords / (W-1)) * 2 - 1
                radial = torch.sqrt(y_norm**2 + x_norm**2)
                pos_encoding[0, c] = radial.repeat(1, W) if c % 2 == 0 else radial.repeat(H, 1)
            else:
                y_norm = (y_coords / (H-1)) * 2 - 1
                x_norm = (x_coords / (W-1)) * 2 - 1
                angle = torch.atan2(y_norm, x_norm) / math.pi
                pos_encoding[0, c] = angle.repeat(1, W) if c % 2 == 0 else angle.repeat(H, 1)
                
        return pos_encoding
    
    def _svd_spatial_compression(self, x_flat, importance=None):
        B, C, N = x_flat.shape
        
        x_t = x_flat.permute(0, 2, 1)  # [B, 4096, 256]
        
        if importance is not None:
            importance_norm = importance.squeeze(1)  # [B, 4096]
            x_t = x_t * importance_norm.unsqueeze(2)  # 加权
        
        svd_compressed_features = []
        
        for b in range(B):
            U, S, V = torch.svd(x_t[b])  # U: [4096, 4096], S: [256], V: [256, 256]
            
            total_energy = torch.sum(S)
            cum_energy = torch.cumsum(S, dim=0)
            k = torch.searchsorted(cum_energy, self.svd_energy_ratio * total_energy).item() + 1
            k = min(k, self.reduced_dim)  # 不超过目标维度
            
            S_selected = S[:k]
            S_enhanced = S_selected * self.svd_value_enhancer(
                S_selected.unsqueeze(1) / S_selected.max()
            ).squeeze(1)
            
            if k < self.reduced_dim:
                U_k = U[:, :k]  # [4096, k]
                U_k_expanded = nn.functional.interpolate(
                    U_k.unsqueeze(0).unsqueeze(0),  # [1, 1, 4096, k]
                    size=(N, self.reduced_dim),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)  # [4096, reduced_dim]

                compressed = U_k_expanded @ torch.diag(
                    torch.cat([S_enhanced, torch.zeros(self.reduced_dim - k, device=x_t.device)])
                )
            else:
                U_k = U[:, :self.reduced_dim]  # [4096, reduced_dim]
                compressed = U_k @ torch.diag(S_enhanced[:self.reduced_dim])
            
            compressed_t = compressed.T  # [reduced_dim, 4096]
            
            pool_size = N // self.reduced_dim  # 4096/256 = 16
            compressed_pooled = compressed_t.reshape(self.reduced_dim, self.reduced_dim, pool_size).mean(dim=2)  # [reduced_dim, reduced_dim]
            
            V_k = V[:, :self.reduced_dim] if V.shape[1] >= self.reduced_dim else V
            channel_info = V_k.T  # [reduced_dim, C] 或 [V.shape[1], C]
            
            if channel_info.shape[0] == self.reduced_dim:
                final_feature = compressed_pooled @ channel_info  # [reduced_dim, C]
            else:
                channel_proj = nn.Linear(channel_info.shape[0], self.reduced_dim, device=x_t.device)
                channel_info_proj = channel_proj(channel_info.T).T  # [reduced_dim, C]
                final_feature = compressed_pooled @ channel_info_proj
            
            svd_compressed_features.append(final_feature.T)  # [C, reduced_dim]
        
        return torch.stack(svd_compressed_features, dim=0)  # [B, C, reduced_dim]
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        channel_weights = self.channel_attention(x)  # [B, C, 1, 1]
        x_channel = x * channel_weights
        
        x_pos = x_channel + self.pos_encoding
        
        importance_map = self.importance_estimator(x_pos)  # [B, 1, H, W]
        
        x_res = self.residual_proj(x_pos)
        
        x_flat = x_pos.flatten(2)  # [B, C, H*W] = [B, 256, 4096]
        importance_flat = importance_map.flatten(2)  # [B, 1, 4096]
        
        svd_features = self._svd_spatial_compression(x_flat, importance_flat)  # [B, 256, 256]
        
        svd_fused = self.spatial_fusion(svd_features.permute(0, 2, 1)).permute(0, 2, 1)
        
        residual_global = x_res.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        residual_global = residual_global.expand(-1, -1, H, W).flatten(2)  # [B, C, 4096]

        residual_pooled = nn.functional.adaptive_avg_pool1d(residual_global, self.reduced_dim)  # [B, C, 256]

        output = svd_fused + residual_pooled * 0.3
  
        output = nn.functional.layer_norm(output, [C, self.reduced_dim])
        
        return output


def top_k_top_p_filtering(
    logits, top_k = 0, top_p = 1.0,
    filter_value = -float("Inf"),
    min_tokens_to_keep = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        mask[config.n_unmasked: config.n_unmasked+1, config.n_unmasked: config.n_unmasked+1] = 1
        mask[config.n_unmasked+1: config.n_unmasked+5, config.n_unmasked+1: config.n_unmasked+5] = 1
        mask[config.n_unmasked+5: config.n_unmasked+14, config.n_unmasked+5: config.n_unmasked+14] = 1
        mask[config.n_unmasked+14: config.n_unmasked+30, config.n_unmasked+14: config.n_unmasked+30] = 1
        mask[config.n_unmasked+30: config.n_unmasked+55, config.n_unmasked+30: config.n_unmasked+55] = 1
        mask[config.n_unmasked+55: config.n_unmasked+91, config.n_unmasked+55: config.n_unmasked+91] = 1
        mask[config.n_unmasked+91: config.n_unmasked+140, config.n_unmasked+91: config.n_unmasked+140] = 1
        mask[config.n_unmasked+140: config.n_unmasked+204, config.n_unmasked+140: config.n_unmasked+204] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present   # TODO: check that this does not break anything


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        self.n_unmasked = n_unmasked
        # input embedding stem
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.emb_emb = PositionAwareSVDSpatialReduction()
        self.drop = nn.Dropout(config.embd_pdrop)
        self.word_embed = nn.Linear(32, 256)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.word_embed(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            embeddings = self.emb_emb(embeddings).permute(0, 2, 1) # B hw C
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        logits = logits[:, self.n_unmasked-1:]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
