
import torch

from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F



def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class CrossPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, *args, **kwargs):

        norm_x = self.norm(x)
        norm_x2 = self.norm(x2)
        return self.fn(norm_x, norm_x2, *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def CrossAggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding=1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride=2, padding=1)
    )





class AbsolutePositionalCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., gamma=0.4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.gamma = gamma


        positions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        dist = torch.cdist(positions, positions, p=2)


        self.M = (dist <= 1.0).float()


        self.P = torch.zeros(4, 4)
        for j in range(4):
            for i in range(4):
                if dist[i, j] > 1.0:
                    self.P[i, j] = -(j+1) * gamma


        self.register_buffer('M_mask', self.M)
        self.register_buffer('P_matrix', self.P)


        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        '''
        We use linear layers to implement the 1×1 convolution in the figure of the paper, and the two are equivalent.
        '''

    def forward(self, x, context):


        h = self.heads


        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)


        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))


        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale


        attn = attn * self.M_mask
        attn = attn + self.P_matrix
        attn = attn.softmax(dim=-1)
        attn = attn * self.M_mask


        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossTransformer(nn.Module):
    def __init__(self, dim,seq_len, depth, heads, mlp_mult, gamma,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                AbsolutePositionalCrossAttention(dim, heads=heads, dropout=dropout,gamma=gamma),
                nn.Sequential(
                    nn.Linear(dim, dim * mlp_mult),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * mlp_mult, dim),
                    nn.Dropout(dropout)
                )
            ]))
    def forward(self, x, x2):

        b, d, _, _ = x.shape
        x_seq = rearrange(x, 'b d h w -> b (h w) d')
        x2_seq = rearrange(x2, 'b d h w -> b (h w) d')

        for norm, attn, ff in self.layers:

            x_norm = norm(x_seq)
            x2_norm = norm(x2_seq)


            x_seq = x_seq + attn(x_norm, x2_seq)
            x2_seq = x2_seq + attn(x2_norm, x_seq)


            x_seq = ff(x_seq) + x_seq
            x2_seq = ff(x2_seq) + x2_seq


        x_out = rearrange(x_seq, 'b (h w) d -> b d h w', h=2, w=2)
        x2_out = rearrange(x2_seq, 'b (h w) d -> b d h w', h=2, w=2)

        return x_out, x2_out


class SpatialTemporalCausalAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., gamma=0.4):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.gamma = gamma


        positions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        dist = torch.cdist(positions, positions, p=2)


        self.M = (dist <= 1.0).float()


        self.P = torch.zeros(4, 4)
        for j in range(4):
            for i in range(4):
                if dist[i, j] > 1.0:
                    self.P[i, j] = -(j+1) * gamma


        self.register_buffer('M_mask', self.M)
        self.register_buffer('P_matrix', self.P)

        '''
        We use linear layers to implement the 1×1 convolution in the figure of the paper, and the two are equivalent.
        '''
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )


        self.temp_to_q = nn.Linear(dim, dim)
        self.temp_to_kv = nn.Linear(dim, dim * 2)
        self.temp_to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def spatial_attention(self, x):

        h = self.heads


        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)


        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)


        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale


        attn = attn * self.M_mask
        attn = attn + self.P_matrix
        attn = attn.softmax(dim=-1)
        attn = attn * self.M_mask


        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def temporal_attention(self, x1, x2):



        h = self.heads


        q = self.temp_to_q(x2)

        kv = self.temp_to_kv(x1)
        k, v = kv.chunk(2, dim=-1)


        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)


        kv_self = self.temp_to_kv(x2)
        k_self, v_self = kv_self.chunk(2, dim=-1)

        k_self = rearrange(k_self, 'b n (h d) -> b h n d', h=h)
        v_self = rearrange(v_self, 'b n (h d) -> b h n d', h=h)


        k_cat = torch.stack([k, k_self], dim=3)
        v_cat = torch.stack([v, v_self], dim=3)


        attn_score = torch.einsum('b h n d, b h n k d -> b h n k', q, k_cat) * self.scale
        attn_weights = F.softmax(attn_score, dim=-1)


        out = torch.einsum('b h n k, b h n k d -> b h n d', attn_weights, v_cat)  # [B, h, N, D]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.temp_to_out(out)

    def forward(self, x1, x2):

        x1 = self.spatial_attention(x1)+x1
        x2 = self.spatial_attention(x2)+x2


        x2 = self.temporal_attention(x1, x2) + x2

        return x1, x2


class CausalRelationMining(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, y2_for, y2_back):


        h = self.heads


        q = self.to_q(y2_for)
        k = self.to_k(y2_back)
        v = self.to_v(y2_back+y2_for)


        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)


        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)


        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')


        out = out + y2_for + y2_back
        return self.to_out(out)


class CausalAttentionBlock(nn.Module):
    def __init__(self, dim, depth, heads, mlp_mult, gamma,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),  # 归一化层
                SpatialTemporalCausalAttention(dim, heads=heads, dropout=dropout,gamma=gamma),  # STCA
                nn.Sequential(  # FFN
                    nn.Linear(dim, dim * mlp_mult),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * mlp_mult, dim),
                    nn.Dropout(dropout)
                ),
                CausalRelationMining(dim, heads=heads, dropout=dropout)  # CRM
            ]))

    def forward(self, x1, x2):

        y1_for = x1
        y2_for = x2
        y1_back=x2
        y2_back=x1
        for norm, st_attn, ff, crm in self.layers:

            y1_for_norm = norm(y1_for)
            y2_for_norm = norm(y2_for)


            y1_for, y2_for = st_attn(y1_for_norm, y2_for_norm)


            y1_for = ff(y1_for) + y1_for
            y2_for = ff(y2_for) + y2_for


            y1_back_norm = norm(y1_back)
            y2_back_norm = norm(y2_back)


            y1_back, y2_back = st_attn(y1_back_norm, y2_back_norm)


            y1_back = ff(y1_back) + y1_back
            y2_back = ff(y2_back) + y2_back


            y_long = crm(y2_for, y2_back)


        y_all = torch.cat([y_long, y1_for, y1_back], dim=1)  # [b, 12, d]


        y_all = rearrange(y_all, 'b (t n) d -> b t n d', n=4)
        return y_all
class CausalNet(nn.Module):
    def  __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
            gamma=0.4
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2 #
        fmap_size = image_size // patch_size #
        blocks = 2 ** (num_hierarchies - 1)#

        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))


        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        self.to_patch_embedding1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )


        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        self.layers1 = nn.ModuleList([])

        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat

            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))
            self.layers1.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))





        self.CrossAttention=CrossTransformer(256, seq_len, depth, 8, mlp_mult, gamma,dropout)

        self.cab = CausalAttentionBlock(
            dim=256,
            depth=1,
            heads=8,
            mlp_mult=mlp_mult,
            dropout=dropout,
            gamma=gamma
        )


        self.mlp_head = nn.Sequential(

            nn.Linear(1536 * 2, 2048),
            nn.Linear(2048, num_classes)
        )
    def forward(self, img):
        img0=img[:,0,:,:,:]
        x = self.to_patch_embedding(img0)
        num_hierarchies = len(self.layers)
        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)

        img1 = img[:, 1, :, :, :]
        x1 = self.to_patch_embedding(img1)
        num_hierarchies1 = len(self.layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies1)), self.layers):
            block_size = 2 ** level
            x1 = rearrange(x1, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)
            x1 = transformer(x1)
            x1 = rearrange(x1, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)
            x1 = aggregate(x1)

        img2 = img[:, 2, :, :, :]

        x2 = self.to_patch_embedding1(img2)

        num_hierarchies2 = len(self.layers1)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies2)), self.layers1):
            block_size = 2 ** level
            x2 = rearrange(x2, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)
            x2 = transformer(x2)
            x2 = rearrange(x2, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)
            x2 = aggregate(x2)

        img3 = img[:, 3, :, :, :]

        x3 = self.to_patch_embedding1(img3)

        num_hierarchies2 = len(self.layers1)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies2)), self.layers1):
            block_size = 2 ** level
            x3 = rearrange(x3, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)

            x3 = transformer(x3)

            x3 = rearrange(x3, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)

            x3 = aggregate(x3)

        b,d,w,h=x.shape
        x=x.reshape(b,d//4,2,2)
        x1 = x1.reshape(b, d // 4, 2, 2)
        x2 = x2.reshape(b, d // 4, 2, 2)

        x3 = x3.reshape(b, d // 4, 2, 2)
        x2,x3=self.CrossAttention(x2,x3)

        x=x+x2
        x1=x1+x3

        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        x1_seq = rearrange(x1, 'b c h w -> b (h w) c')


        cab_out = self.cab(x_seq, x1_seq).flatten(1)  # [b, 3, 4, c]


        return self.mlp_head(cab_out)


