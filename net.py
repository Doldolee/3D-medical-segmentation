import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
import torch

class VNet(nn.Module):
    def __init__(self, in_ch = 1, num_class=2):
        super(VNet, self).__init__()

        self.en1 = BigBlock(depth=1, in_ch=1, out_ch=16)
        self.exp_ch1 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.down1 = nn.Conv3d(16, 32, kernel_size=2, stride=2)

        self.en2 = BigBlock(depth=2, in_ch=32, out_ch=32)
        self.exp_ch2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv3d(32, 64, kernel_size=2, stride=2)

        self.en3 = BigBlock(depth=3, in_ch=64, out_ch=64)
        self.exp_ch3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.down3 = nn.Conv3d(64, 128, kernel_size=2, stride=2)

        self.en4 = BigBlock(depth=3, in_ch=128, out_ch=128)
        self.exp_ch4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.down4 = nn.Conv3d(128, 256, kernel_size=2, stride=2)

        self.en5 = BigBlock(depth=3, in_ch=256, out_ch=256)
        self.up5 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)

        self.de4 = BigBlock(depth=3, in_ch=256, out_ch=256)
        self.up4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)

        self.de3 = BigBlock(depth=3, in_ch=128, out_ch=128)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        self.de2 = BigBlock(depth=2, in_ch=64, out_ch=64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)

        self.de1 = BigBlock(depth=1, in_ch=32, out_ch=32)

        self.out_conv = nn.Conv3d(32, num_class, 1, 1)

    def forward(self, x):
        en1_res = x
        en1 = self.en1(x)
        en1 += en1_res

        en2_res = self.down1(en1)
        en2 = self.en2(en2_res)
        en2 += en2_res

        en3_res = self.down2(en2)
        en3 = self.en3(en3_res)
        en3 += en3_res

        en4_res = self.down3(en3)
        en4 = self.en4(en4_res)
        en4 += en4_res

        en5_res = self.down4(en4)
        en5 = self.en5(en5_res)
        en5 += en5_res

        de4_res = self.up5(en5)
        en4 = self.exp_ch4(en4)
        de4 = de4_res + en4
        de4 = self.de4(de4)
        de4 += de4_res

        de3_res = self.up4(en4)
        en3 = self.exp_ch3(en3)
        de3 = de3_res + en3
        de3 = self.de3(de3)
        de3 += de3_res

        de2_res = self.up3(en3)
        en2 = self.exp_ch2(en2)
        de2 = de2_res + en2
        de2 = self.de2(de2)
        de2 += de2_res

        de1_res = self.up2(en2)
        en1 = self.exp_ch1(en1)
        de1 = de1_res + en1
        de1 = self.de1(de1)
        de1 += de1_res

        output = self.out_conv(de1)

        return output        
    
    
class UNETR(nn.Module):
    def __init__(self, img_shape=(224, 224, 224), input_dim=3, output_dim=3, 
                 embed_dim=768, patch_size=16, num_heads=8, dropout=0.1, light_r=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]
        self.conv_channels = [int(i/light_r) for i in [32, 64, 128, 256, 512, 1024]]

        self.embedding = Embeddings((input_dim,*img_shape))
        
        # Transformer Encoder
        self.transformer = \
            TransformerBlock(
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, self.conv_channels[0], 3),
                Conv3DBlock(self.conv_channels[0], self.conv_channels[1], 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, self.conv_channels[2]),
                Deconv3DBlock(self.conv_channels[2], self.conv_channels[2]),
                Deconv3DBlock(self.conv_channels[2], self.conv_channels[2])
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, self.conv_channels[3]),
                Deconv3DBlock(self.conv_channels[3], self.conv_channels[3]),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, self.conv_channels[4])

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, self.conv_channels[4])

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.conv_channels[5], self.conv_channels[3]),
                Conv3DBlock(self.conv_channels[3], self.conv_channels[3]),
                Conv3DBlock(self.conv_channels[3], self.conv_channels[3]),
                SingleDeconv3DBlock(self.conv_channels[3], self.conv_channels[3])
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.conv_channels[4], self.conv_channels[2]),
                Conv3DBlock(self.conv_channels[2], self.conv_channels[2]),
                SingleDeconv3DBlock(self.conv_channels[2], self.conv_channels[2])
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.conv_channels[3], self.conv_channels[1]),
                Conv3DBlock(self.conv_channels[1], self.conv_channels[1]),
                SingleDeconv3DBlock(self.conv_channels[1], self.conv_channels[1])
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(self.conv_channels[2], self.conv_channels[1]),
                Conv3DBlock(self.conv_channels[1], self.conv_channels[1]),
                SingleConv3DBlock(self.conv_channels[1], output_dim, 1)
            )

    def forward(self, x):
        z0 = x
        x = self.embedding(x)
        z = self.transformer(x)
        z3, z6, z9, z12 = z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output
    

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    
    
class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)    

class Embeddings(nn.Module):
    def __init__(self, input_shape, patch_size=16, embed_dim=768, dropout=0.):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = input_shape[-4]
        self.n_patches = int((input_shape[-1] * input_shape[-2] * input_shape[-3]) / (patch_size * patch_size * patch_size))
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=self.in_channels, out_channels=self.embed_dim,
                                          kernel_size=self.patch_size, stride=self.patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, self.embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = rearrange(x, "b n h w d -> b (h w d) n")
        # batch, embed_dim, height/patch, width/patch, depth/patch
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, depth=12, dropout=0., extract_layers=[3,6,9,12]):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embed_dim, MultiHeadAttention(embed_dim, num_heads, dropout)),
                PreNorm(embed_dim, FeedForwardBlock(embed_dim, expansion=4))
            ]))            
        self.extract_layers = extract_layers
        
    def forward(self, x):
        extract_layers = []
        
        for cnt, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if cnt+1 in self.extract_layers:
                extract_layers.append(x)
            
        return extract_layers


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(Conv3dBlock, self).__init__()
        self.net = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding),
                                  nn.BatchNorm3d(out_ch),
                                  nn.PReLU()
                                )
    def forward(self, x):
        return self.net(x)

class BigBlock(nn.Module):
    def __init__(self, depth, in_ch, out_ch):
        super(BigBlock, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Conv3dBlock(in_ch, out_ch))
            in_ch = out_ch

    def forward(self, x):
        for i in self.layers:
            x = i(x)
        return x