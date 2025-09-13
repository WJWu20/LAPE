from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from models.Block.Blocks import Block
from models.text_encoder import CLIPTextEncoder
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np


class CoOpPromptLearner(nn.Module):
    def __init__(self, emb_dim=768, n_ctx=4, class_specific=False, num_classes=1, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_ctx = n_ctx
        self.class_specific = class_specific
        self.num_classes = num_classes

        if class_specific:
            self.ctx = nn.Parameter(torch.randn(num_classes, n_ctx, emb_dim) * 0.02)
        else:
            self.ctx = nn.Parameter(torch.randn(n_ctx, emb_dim) * 0.02)

        self.fuse = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim)
        )

        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, text_feat):
        B, N, D = text_feat.shape
        ctx_mean = self.ctx.mean(dim=0, keepdim=True).expand(B, N, D)

        fused = self.fuse(torch.cat([ctx_mean, text_feat], dim=-1))
        out = text_feat + torch.tanh(self.gate) * fused
        return out


class TextAdapter(nn.Module):
    def __init__(self, in_dim=512, out_dim=768, dropout=0.1):
        super(TextAdapter, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))           # -> (B, 1, 768)
        x = self.dropout(self.norm(self.act(self.fc2(x))))      # LayerNorm
        x = self.fc3(x)
        return x


class SupervisedMAE(nn.Module):
    """ CntVit with VisionTransformer backbone
    """

    def __init__(self, img_size=384, K=16, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path_rate=0,
                 text_pretrained='/media/lcc/DATA/wwj/path/ViT-B-16.pt'):
        super().__init__()
        ## Setting the model
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        ## Global Setting
        self.patch_size = patch_size
        self.img_size = img_size
        self.K = K
        self.norm_pix_loss = norm_pix_loss
        ## Global Setting

        ## Encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.norm = norm_layer(embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

        self.v_y = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        self.density_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_norm = norm_layer(decoder_embed_dim)
        ### decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        ### decoder blocks
        ## Decoder specifics
        ## Regressor
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(513, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )
        ## Regressor

        ## text encoder
        self.text_encoder = CLIPTextEncoder(
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
            embed_dim=512,
            pretrained=text_pretrained
        )
        self.text_proj = TextAdapter()

        self.coop = CoOpPromptLearner(
            emb_dim=768, n_ctx=4, class_specific=False, num_classes=1, dropout=0.1
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.text_encoder.init_weights()

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def text_embedding(self, texts):
        text_embeddings = self.text_encoder(texts.to('cuda'))
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

    def forward_encoder(self, x, y):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        _, l, d = x.shape
        attns = []
        x_y = torch.cat((x, y), dim=1)
        for i, blk in enumerate(self.blocks):
            x_y, attn = blk(x_y)
            attns.append(attn)
        x_y = self.norm(x_y)
        x = x_y[:, :l, :]
        y = x_y[:, l:, :]
        return x, y, attns

    def forward_decoder(self, x, y, text):
        x = self.decoder_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed
        b, l_x, d = x.shape
        y_embed = self.decoder_embed(y)
        t_embed = self.decoder_embed(text)
        x = torch.cat((x, y_embed, t_embed), dim=1)
        attns = []
        xs = []
        ys = []
        for i, blk in enumerate(self.decoder_blocks):
            x, attn = blk(x)
            if i == 2:
                x = self.decoder_norm(x)
            attns.append(attn)
            xs.append(x[:, :l_x, :])
            ys.append(x[:, l_x:-1, :])
        return xs, ys, attns

    def extract_topk_vis(self, vis_feat, attns, l=576):

        attn = attns[-1].mean(1)[:, l:, :l].sum(1)

        topk_values, topk_indices = torch.topk(attn, k=self.K, dim=-1)

        B, N, D = vis_feat.shape
        _, K = topk_indices.shape

        index_expanded = topk_indices.unsqueeze(-1).expand(B, K, D)

        selected_feat = vis_feat.gather(1, index_expanded)

        return selected_feat

    def AttentionEnhance(self, attns, l=24):
        l_x = int(l * l)
        l_y = int(self.K + 2)
        r = self.img_size // self.patch_size
        attns = torch.mean(attns, dim=1)

        attns_x2y = attns[:, l_x:, :l_x]
        attns_x2y = rearrange(attns_x2y, 'b (n ly) l->b n ly l', ly=l_y)
        attns_x2y = attns_x2y.sum(2)

        attns_x2y = torch.mean(attns_x2y, dim=1).unsqueeze(-1)
        attns_x2y = rearrange(attns_x2y, 'b (w h) c->b c w h', w=r, h=r)
        return attns_x2y

    def MacherMode(self, xs, ys, attn):
        x = xs[-1]
        B, L, D = x.shape
        y = ys[-1]
        B, Ly, D = y.shape
        density_feature = rearrange(x, 'b (w h) d->b d w h', w=24)
        density_enhance = self.AttentionEnhance(attn[-1], l=int(np.sqrt(L)))
        density_feature2 = torch.cat((density_feature.contiguous(), density_enhance.contiguous()), dim=1)

        return density_feature2

    def Regressor(self, feature):
        feature = F.interpolate(
            self.decode_head0(feature), size=feature.shape[-1] * 2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
            self.decode_head1(feature), size=feature.shape[-1] * 2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
            self.decode_head2(feature), size=feature.shape[-1] * 2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
            self.decode_head3(feature), size=feature.shape[-1] * 2, mode='bilinear', align_corners=False)
        feature = feature.squeeze(-3)
        return feature

    def forward(self, samples):
        imgs = samples[0]
        text = samples[1]
        description = samples[2]
        text = self.text_proj(torch.cat([self.text_encoder(text).unsqueeze(-2), self.text_encoder(description).unsqueeze(-2)], dim=1))
        text = self.coop(text)
        latent, text, attns = self.forward_encoder(imgs, text)
        y_latent = self.extract_topk_vis(latent, attns)
        xs, ys, attns = self.forward_decoder(latent, y_latent, text)
        density_feature = self.MacherMode(xs, ys, attns)
        density_map = self.Regressor(density_feature)

        return density_map


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
