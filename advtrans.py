import torch
import torch.nn as nn

from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange

#输入图片patch(CNN) and flatten
#图片为cifar10数据集3*32*32
class PatchEmbedding(nn.Module):
    def __init__(self,img_size=32,patch_size=16,in_c=3,embed_dim=16*16*3):
        super().__init__()
        
        img_size = (img_size,img_size) 
        #patch_size = (patch_size,patch_size) 
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size,img_size[1]//patch_size) 
        self.patches = self.grid_size[0]*self.grid_size[1]

        #图片为3*32*32，划分为16*16*3的小块，共self.patches=4
        #利用einops对图片patch并重新排列
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = self.patch_size, p2 = self.patch_size),
            nn.Linear(embed_dim, embed_dim),
        )
        #self.proj = nn.Conv2d(in_c,embed_dim,kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        B,C,H,W = x.shape
    
        assert H == self.img_size[0] and W == self.img_size[1],\
            f"Input image size ({H}*{W}) doesn't match mode ({self.img_size[0]}*{self.img_size[1]})."
    
        x = self.to_patch_embedding(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) #work with diff dim tensor, not just 2D convNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self,dim,num_heads,qkv_bias=False,qk_scale=None,attn_drop_ratio=0.,\
        proj_drop_ratio=0.2,window_size=2):
        # dim:输入token的dim
        # num_heads:头数
        # qkv_bias：生成qkv的bias
        # qk_scale: None时使用：根号dk分之一
        super(Attention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads #每个头分到的qkv的个数
        self.scale = qk_scale or head_dim**(-0.5)
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.window_size = window_size

        if self.window_size != 0:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2 ；contiguous()函数的作用：把tensor变成在内存中连续分布的形式。
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self,x):
        B,N,C=x.shape
        #B:batchsize,N=1+num_patches(1:class_token),C:embed_dim

        qkv = self.qkv(x)
        qkv = qkv.reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]

        attn = (q @ k.transpose(-2,-1))*self.scale
        if self.window_size != 0:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1).clone()].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1) #最后一维
        attn_map = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x,attn_map

class MLP(nn.Module):
    def __init__(self,in_feature,hidde_feature=None,out_feature=None,act_layer=nn.GELU,drop=0.2):
        super().__init__()

        out_feature = out_feature or in_feature
        hidde_feature = hidde_feature or in_feature

        self.fc1 = nn.Linear(in_feature,hidde_feature)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidde_feature,out_feature)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class En_Block(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio=4.,qkv_bias=False,qk_scale=None,attn_drop_ratio=0.,\
        proj_drop_ratio=0.2,act_layer=nn.GELU,drop_path_ratio=0.,norm_layer=nn.LayerNorm,window_size=2):
        # dim:输入token的dim
        # num_heads:头数
        # mlp_ratio: mlp第一个FC节点数是输入的4倍
        # qkv_bias：生成qkv的bias
        # qk_scale: None时使用：根号dk分之一
        # act_layer: 激活函数
        # norm_layer: 归一化函数
        super(En_Block,self).__init__()
        
        self.norm = norm_layer(dim)
        self.attn = Attention(dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,\
            attn_drop_ratio=attn_drop_ratio,proj_drop_ratio=proj_drop_ratio,window_size=window_size)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio>0. else nn.Identity()
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_feature=dim, hidde_feature=mlp_hidden_dim,act_layer=act_layer,\
            drop=proj_drop_ratio)
        
    def forward(self,x): #残差结构
        x1= x
        x = self.norm(x)
        x,attnmap = self.attn(x)
        x = self.drop_path(x)
        x = x1+x
        y=x

        x2= x
        x = self.norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x2+x

        global attn
        attn = attnmap
        return x

class PatchSplicing(nn.Module):
    def __init__(self,img_size=32,patch_size=16,in_c=3):
        super(PatchSplicing,self).__init__()
        
        self.patch = img_size//patch_size
        self.c = in_c
        self.patch_size = patch_size
        self.to_patch_splicing = nn.Sequential(
            Rearrange('b (h w) (c p1 p2) -> b  c (h p1) (w p2)', h= self.patch, w =self.patch, c=self.c,
            p1 = self.patch_size, p2 = self.patch_size),
            nn.Linear(img_size, img_size),
        )

    def forward(self,x):
        x = self.to_patch_splicing(x)
        return x


def _init_AT_weight(m):
    if isinstance(m,nn.Linear):
        nn.init.trunc_normal_(m.weight,std=0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m,nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class adv_transformer(nn.Module):
    def __init__(self,img_size=32,patch_size=16,in_c=3,num_classes=10,#cifar10
        embed_dim=768,depth=2,num_heads=1,mlp_ratio=2.,
        qkv_bias=True,qk_scale=None,
        proj_drop_ratio=0.5,attn_drop_ratio=0.5,drop_path_ratio=0.,
        embed_layer=PatchEmbedding,norm_layer=None,act_layer=None, patch_splicing_layer=PatchSplicing):
        super(adv_transformer,self).__init__()
        
        self.num_classes = num_classes
        self.num_feature = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm,eps=1e-6) # partial默认固定函数的某些参数
        act_layer = act_layer or nn.GELU

        # patch 
        self.patch = embed_layer(img_size=img_size,patch_size=patch_size,in_c=in_c,embed_dim=embed_dim)
        num_patches = self.patch.patches
        window_size = int(num_patches**0.5)
        #position embed and init
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dim))
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        #dropout
        self.pos_drop = nn.Dropout(p=proj_drop_ratio)
        #depth次encoder block-->生成长度为depth的drop_ratio,取值区间为(0,drop_path_ratio)
        dpr = [x.item() for x in torch.linspace(0,drop_path_ratio,depth)]
        #depth个encoder block
        self.blocks = nn.Sequential(*[
            En_Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,proj_drop_ratio=proj_drop_ratio,
            act_layer=nn.GELU,drop_path_ratio=dpr[i],norm_layer=nn.LayerNorm,window_size=window_size)
            for i in range(depth)
        ])
        #图片重新拼接
        self.patchsplicing = patch_splicing_layer(img_size=img_size,patch_size=patch_size,in_c=in_c)
        #权重初始化
        self.apply(_init_AT_weight)
        
    def forward(self,x):
        #patch
        x = self.patch(x)
        x3=x
        #pos
        x = x+self.pos_embed
        x2=x
        #tran encoder
        x = self.blocks(x)
        x1 = x
        #splicing
        x = self.patchsplicing(x)
        
        return x