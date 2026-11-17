import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import cfg

Z_MIN_MM = -200.0
Z_MAX_MM =  200.0
Z_BIN_MM =    2.0       # 2 mm → 201 bins
Z_SIGMA_B  =    5.0     # Gaussian sigma in *bins*


def transform_keypoints_batched(kpts: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """
    kpts:  [B, 42, 2]
    trans: [B, 2, 3]  (affine per batch)
    returns: [B, 42, 2]
    """
    B, N, _ = kpts.shape
    ones = torch.ones(B, N, 1, dtype=kpts.dtype, device=kpts.device)
    kpts_h = torch.cat([kpts, ones], dim=-1)          # [B, N, 3]
    return torch.einsum('bnj,bij->bni', kpts_h, trans)  # <-- no transpose


# ---- Z (depth) SimCC helpers ----


def backproject_xy_mm(kps_xy, Z_abs_mm, focal, princpt):
    # kps_xy: (B,K,2) pixels; Z_abs_mm: (B,K) mm; focal/princpt: (2,) or (B,2)
    u, v = kps_xy[..., 0], kps_xy[..., 1]
    if focal.ndim == 1:
        fx = kps_xy.new_full(u.shape, float(focal[0]))
        fy = kps_xy.new_full(v.shape, float(focal[1]))
    else:
        fx = focal[:, 0].unsqueeze(1).expand_as(u)
        fy = focal[:, 1].unsqueeze(1).expand_as(v)

    if princpt.ndim == 1:
        cx = kps_xy.new_full(u.shape, float(princpt[0]))
        cy = kps_xy.new_full(v.shape, float(princpt[1]))
    else:
        cx = princpt[:, 0].unsqueeze(1).expand_as(u)
        cy = princpt[:, 1].unsqueeze(1).expand_as(v)

    X_mm = (u - cx) / fx * Z_abs_mm
    Y_mm = (v - cy) / fy * Z_abs_mm
    return X_mm, Y_mm

def assemble_xyz(kps_xy, z_rel_mm, Z_root_mm_r, Z_root_mm_l, focal, princpt):
    """
    kps_xy:    (B,K,2) pixels
    z_rel_mm:  (B,K) root-relative depth (mm)
    Z_root_mm_r: (B,)  absolute root depth right (mm); can be GT during training
    Z_root_mm_l: (B,)  absolute root depth left (mm); can be GT during training
    focal:     (2,) or (B,2) in pixels
    princpt:   (2,) or (B,2) in pixels
    """
    B, K, _ = kps_xy.shape
    # absolute per-joint depth = root depth + rel depth
    Kh = K // 2

    Zr = Z_root_mm_r.view(B, 1).expand(B, Kh)  # (B, Kh)
    Zl = Z_root_mm_l.view(B, 1).expand(B, Kh)  # (B, Kh)

    Z_root_cat = torch.cat([Zr, Zl], dim=1)    # (B, K) matches z_rel_mm ordering
    Z_abs_mm   = Z_root_cat + z_rel_mm         # (B, K)
    X_mm, Y_mm = backproject_xy_mm(kps_xy, Z_abs_mm, focal, princpt)       # (B,K), (B,K)

    # absolute camera-frame XYZ
    XYZ_abs = torch.stack([X_mm, Y_mm, Z_abs_mm], dim=-1)                  # (B,K,3)

    return XYZ_abs


def build_z_simcc_targets(
    z_mm,                         # (B, K) root-relative depth in mm (e.g., [-200, 200])
    z_min_mm=-200.0,
    z_max_mm= 200.0,
    bin_size_mm=2.0,
    sigma_bins=1.0,               # Gaussian sigma in *bins*, not mm
    keypoint_weights=None
):
    """
    Returns gt_z: (B, K, Zbins) SimCC-style probabilities over depth bins.
    """
    B, K = z_mm.shape
    z_mm = z_mm.clamp(z_min_mm, z_max_mm)
    Zbins = int(round((z_max_mm - z_min_mm) / bin_size_mm)) + 1

    # mm -> bin index (float), then Gaussian label smoothing
    mu_bins = (z_mm - z_min_mm) / bin_size_mm  # (B, K) in bin units
    gt_z = make_simcc_1d(mu_bins, Zbins, sigma=sigma_bins)  # (B, K, Zbins)
    return gt_z, keypoint_weights, (Zbins, z_min_mm, bin_size_mm)

def decode_simcc_z_mm(pred_z, z_min_mm, bin_size_mm, use_argmax=False):
    """
    pred_z: (B, K, Zbins) logits
    returns z_mm: (B, K) in millimeters via expectation (default) or argmax.
    """
    B, K, Zbins = pred_z.shape
    if use_argmax:
        idx = pred_z.argmax(dim=-1).to(pred_z.dtype)                    # (B, K)
    else:
        pz  = F.softmax(pred_z, dim=-1)
        iz  = torch.arange(Zbins, device=pred_z.device, dtype=pred_z.dtype)
        idx = (pz * iz).sum(dim=-1)                                     # (B, K)
    z_mm = z_min_mm + bin_size_mm * idx
    return z_mm


def make_simcc_1d(mu, num_bins, sigma=2.5):
    """
    mu: (B, K) mean in BIN units
    returns: (B, K, num_bins) probability distribution (sums to 1)
    """
    B, K = mu.shape
    i = torch.arange(num_bins, device=mu.device).float()             # (num_bins,)
    # expand to (B, K, num_bins)
    d = (i.view(1, 1, -1) - mu.unsqueeze(-1)) / sigma
    g = torch.exp(-0.5 * d * d)
    g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
    return g

def build_simcc_targets(joint_coord, img_wh=(256, 256), split_ratio=2.0, sigma=2.5, keypoint_weights=None):
    """
    joint_coord: (B, K, 2) in pixels (x,y)
    keypoint_weights: optional (B, K) visibility/validity in {0,1} or [0,1]
    returns gt_x: (B, K, Wx), gt_y: (B, K, Wy), keypoint_weights: (B, K) or None
    """
    W, H = img_wh
    Wx = int(W * split_ratio)
    Wy = int(H * split_ratio)
    x_px = joint_coord[..., 0].clamp(0, W - 1e-6)
    y_px = joint_coord[..., 1].clamp(0, H - 1e-6)

    # to bin space
    mu_x = (x_px * split_ratio).clamp(0, Wx - 1e-6)
    mu_y = (y_px * split_ratio).clamp(0, Wy - 1e-6)

    gt_x = make_simcc_1d(mu_x, Wx, sigma=sigma)  # (B, K, 512)
    gt_y = make_simcc_1d(mu_y, Wy, sigma=sigma)  # (B, K, 512)

    return gt_x, gt_y, keypoint_weights


class ScaleNorm(nn.Module):
    """ScaleNorm: x * (g / ||x||)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize across last dimension
        denom = x.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
        return x * (self.g / denom)

def _make_act(name: str):
    name = (name or 'ReLU').lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'gelu':
        return nn.GELU()
    if name == 'silu' or name == 'swish':
        return nn.SiLU()
    return nn.ReLU(inplace=True)

class RTMCCBlock(nn.Module):
    """
    Lightweight GAU-like block (Transformer-style):
      LN -> MHA -> Dropout -> Residual
      LN -> Gated FFN (GLU) -> Dropout -> Residual
    Operates on sequences of length K with embedding dim=hidden_dims.
    """
    def __init__(self,
                 out_channels: int,       # kept for API symmetry; not used internally
                 in_dim: int,             # = hidden_dims
                 dim: int,                # = hidden_dims (kept for API symmetry)
                 s: int = 8,              # heads
                 expansion_factor: int = 2,
                 dropout_rate: float = 0.0,
                 drop_path: float = 0.0,  # ignored (no stochastic depth here)
                 attn_type: str = 'self-attn',
                 act_fn: str = 'ReLU',
                 use_rel_bias: bool = False,
                 pos_enc: bool = False):
        super().__init__()
        assert attn_type == 'self-attn', "Only self-attn supported in this minimal block."

        self.num_heads = max(1, int(s))
        self.emb_dim = in_dim
        self.dropout_rate = dropout_rate

        self.ln1 = nn.LayerNorm(self.emb_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(self.emb_dim)
        hidden_ffn = self.emb_dim * expansion_factor

        # Gated FFN (GLU-like): proj -> split -> act/sigmoid -> gate -> out proj
        self.ffn_in = nn.Linear(self.emb_dim, 2 * hidden_ffn, bias=True)
        self.act = _make_act(act_fn)
        self.ffn_out = nn.Linear(hidden_ffn, self.emb_dim, bias=True)
        self.drop = nn.Dropout(dropout_rate)

        # (Optional) very light positional bias could be added if desired; omitted here.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        y = self.ln1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop(attn_out)

        # Gated FFN
        y = self.ln2(x)
        y = self.ffn_in(y)                      # (B, K, 2*hidden_ffn)
        a, b = y.chunk(2, dim=-1)
        a = a.contiguous()                      # <- make base tensors
        b = b.contiguous()
        y = self.act(a) * torch.sigmoid(b)      # act is NOT inplace
        y = self.ffn_out(y)
        x = x + self.drop(y)
        return x


class RTMCCHeadLite(nn.Module):
    """
    Minimal RTMCC head: Conv -> flatten -> (ScaleNorm + Linear) -> GAU -> (cls_x, cls_y)
    - in_channels: channels of fused feature (e.g., 256)
    - out_channels: number of keypoints (K)
    - input_size: (W, H) of the *input image* used for SimCC split
    - in_featuremap_size: (Wf, Hf) of the feature map fed into the head
    - simcc_split_ratio: e.g., 2.0 (RTMPose default)
    """
    def __init__(self, in_channels, out_channels, input_size, in_featuremap_size,
                 simcc_split_ratio: float = 2.0, final_layer_kernel_size: int = 1,
                 gau_cfg: dict = dict(
                     hidden_dims=256, s=128, expansion_factor=2,
                     dropout_rate=0., drop_path=0., act_fn='ReLU',
                     use_rel_bias=False, pos_enc=False),
                 z_bins: int = None,           # <--- NEW: number of depth bins, or None to disable z
                 z_detach: bool = False        # <--- NEW: stop-grad on z branch to avoid influencing x/y
                 ):
        super().__init__()
        self.out_channels = out_channels   # expect out_channels = K (num keypoints)
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.z_bins = z_bins
        self.z_detach = z_detach

        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]  # Wf*Hf

        self.final_layer = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2
        )
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False)
        )

        W = int(self.input_size[0] * self.simcc_split_ratio)  # Wx bins
        H = int(self.input_size[1] * self.simcc_split_ratio)  # Wy bins

        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc']
        )

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)

        # ---- NEW: z classifier (1-D SimCC over depth bins)
        if self.z_bins is not None:
            self.cls_z = nn.Linear(gau_cfg['hidden_dims'], self.z_bins, bias=False)
        else:
            self.cls_z = None

    def forward(self, feats_tuple):
        """
        Returns:
            pred_x: (B, K, Wx)
            pred_y: (B, K, Wy)
            pred_z: (B, K, Zbins) or None
        """
        feats = feats_tuple[-1]              # B, C, Hf, Wf
        feats = self.final_layer(feats)      # B, K, Hf, Wf
        feats = torch.flatten(feats, 2)      # B, K, Hf*Wf
        feats = self.mlp(feats)              # B, K, hidden
        feats = self.gau(feats)              # B, K, hidden

        pred_x = self.cls_x(feats)           # B, K, Wx
        pred_y = self.cls_y(feats)           # B, K, Wy

        # ---- NEW: depth branch; optional stop-grad to avoid affecting x/y
        pred_z = None
        if self.cls_z is not None:
            feats_z = feats.detach() if self.z_detach else feats
            pred_z = self.cls_z(feats_z)     # B, K, Zbins

        return pred_x, pred_y, pred_z


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BackboneHRNet(nn.Module):
    """
    HRNet backbone with simple multi-scale fusion to 1/4-resolution.
    Works as a drop-in replacement for your BackboneNet (returns fused feature at stride 4).
    """
    def __init__(self, variant='w32', out_channels=256, use_se=True, pretrained=False):
        super().__init__()
        # model names like 'hrnet_w32', 'hrnet_w48'
        self.hrnet = timm.create_model(f'hrnet_{variant}', pretrained=pretrained, features_only=True, out_indices=(0,1,2,3))
        in_chs = [fi['num_chs'] for fi in self.hrnet.feature_info]  # e.g. [32, 64, 128, 256] for W32
        
        # project all stages to a common channel dim, fuse at the highest resolution (index 0, stride=4)
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_chs])
        self.smooth   = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.use_se   = use_se
        if use_se: self.se = SEBlock(out_channels)

    def forward(self, x):
        feats = self.hrnet(x)               # list: [s=4, s=8, s=16, s=32]
        target_size = feats[0].shape[-2:]   # fuse to the highest-res stream (s=4)
        fused = 0
        for f, lat in zip(feats, self.laterals):
            y = lat(f)
            if y.shape[-2:] != target_size:
                y = F.interpolate(y, size=target_size, mode='bilinear', align_corners=False)
            fused = fused + y
        fused = self.smooth(fused)
        if self.use_se:
            fused = self.se(fused)          # light channel attention on fused map
        target_hw = (cfg.output_hm_shape[1], cfg.output_hm_shape[2])  # e.g., (64, 64)
        if fused.shape[-2:] != target_hw:
            fused = F.interpolate(fused, size=target_hw, mode='bilinear', align_corners=False)
        return fused, feats                 # fused (s=4), plus raw per-stage feats for aux losses


# --- your SEBlock & BackboneHRNet here (unchanged) ---

def simcc_loss(pred_x, pred_y, gt_x, gt_y, keypoint_weights=None):
    """
    KL loss between predicted SimCC logits and target distributions.
    pred_x: (B, K, Wx) logits
    pred_y: (B, K, Wy) logits
    gt_x:   (B, K, Wx) probabilities (sum=1 along -1)
    gt_y:   (B, K, Wy) probabilities (sum=1 along -1)
    keypoint_weights: (B, K) optional visibility/validity weights
    """
    log_px = F.log_softmax(pred_x, dim=-1)
    log_py = F.log_softmax(pred_y, dim=-1)

    # elementwise KL (no reduction yet): same shape as inputs
    loss_x = F.kl_div(log_px, gt_x, reduction='none')  # (B, K, Wx)
    loss_y = F.kl_div(log_py, gt_y, reduction='none')  # (B, K, Wy)

    # average across bins
    loss_x = loss_x.mean(dim=-1)  # (B, K)
    loss_y = loss_y.mean(dim=-1)  # (B, K)
    loss = 0.5 * (loss_x + loss_y)  # (B, K)

    if keypoint_weights is not None:
        loss = loss * keypoint_weights

    return loss.mean()

def decode_simcc(pred_x, pred_y, split_ratio: float, img_wh=None, use_argmax: bool=False):
    """
    pred_x: (B, K, Wx) logits
    pred_y: (B, K, Wy) logits
    returns kps_xy: (B, K, 2) in pixel coords
    """
    B, K, Wx = pred_x.shape
    Wy = pred_y.shape[-1]

    if use_argmax:
        x_bins = pred_x.argmax(dim=-1).to(pred_x.dtype)  # (B, K)
        y_bins = pred_y.argmax(dim=-1).to(pred_y.dtype)  # (B, K)
    else:
        px = F.softmax(pred_x, dim=-1)  # (B, K, Wx)
        py = F.softmax(pred_y, dim=-1)  # (B, K, Wy)
        ix = torch.arange(Wx, device=pred_x.device, dtype=pred_x.dtype)  # (Wx,)
        iy = torch.arange(Wy, device=pred_y.device, dtype=pred_y.dtype)  # (Wy,)
        x_bins = (px * ix).sum(dim=-1)  # (B, K)
        y_bins = (py * iy).sum(dim=-1)  # (B, K)

    x_px = x_bins / split_ratio
    y_px = y_bins / split_ratio
    kps_xy = torch.stack([x_px, y_px], dim=-1)  # (B, K, 2)

    if img_wh is not None:
        W, H = img_wh
        kps_xy[..., 0].clamp_(0, W - 1e-6)
        kps_xy[..., 1].clamp_(0, H - 1e-6)

    return kps_xy

class SimCCShift1D(nn.Module):
    @staticmethod
    def shift_logits(logits: torch.Tensor, shift_bins: torch.Tensor) -> torch.Tensor:
        B, K, Bins = logits.shape
        idx = torch.arange(Bins, device=logits.device, dtype=logits.dtype).view(1,1,Bins)
        src  = idx - shift_bins.unsqueeze(-1)
        src0 = src.floor().clamp(0, Bins-1)
        w    = (src - src0).clamp(0, 1)
        src1 = (src0 + 1).clamp(0, Bins-1)

        arngB = torch.arange(B, device=logits.device).view(B,1,1).expand(B,K,Bins)
        arngK = torch.arange(K, device=logits.device).view(1,K,1).expand(B,K,Bins)
        l0 = logits[arngB, arngK, src0.long()]
        l1 = logits[arngB, arngK, src1.long()]
        return (1 - w) * l0 + w * l1

class DatasetJointShiftEmb(nn.Module):
    def __init__(self, num_datasets, K, ref_dataset_id=1,
                 emb_dim=16, hidden=32, out_dim=2):  # out_dim=2 for xy, 3 if you want z too
        super().__init__()
        self.D, self.K, self.out_dim = num_datasets, K, out_dim
        self.ref_id = ref_dataset_id
        self.emb_ds = nn.Embedding(self.D, emb_dim)
        self.emb_k  = nn.Embedding(self.K, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )
        with torch.no_grad():
            self.emb_ds.weight[self.ref_id].zero_()
        self.emb_ds.weight.register_hook(self._zero_ref_hook)
        self.register_buffer('mask_ref', torch.ones(self.D, 1))
        self.mask_ref[self.ref_id] = 0.0

    def _zero_ref_hook(self, g):
        g[self.ref_id].zero_(); return g

    def forward(self, ds_id):
        B = ds_id.size(0)
        e_ds = self.emb_ds(ds_id).unsqueeze(1)                          # (B,1,E)
        e_k  = self.emb_k(torch.arange(self.K, device=ds_id.device)).unsqueeze(0)  # (1,K,E)
        shift = self.mlp(e_ds + e_k)                                     # (B,K,out_dim)
        shift = shift * self.mask_ref[ds_id].view(B,1,1)                 # zero for ref
        # pad to 3 for uniform unpacking
        if self.out_dim == 2:
            pad = torch.zeros(B, self.K, 1, device=shift.device, dtype=shift.dtype)
            shift = torch.cat([shift, pad], dim=-1)
        dx, dy, dz = shift[...,0], shift[...,1], shift[...,2]
        return dx, dy, dz

    @torch.no_grad()
    def add_datasets(self, n_new: int):
        """Grow to accommodate new dataset IDs."""
        if n_new <= 0: return
        newD = self.D + n_new
        # extend ds embedding
        old_weight = self.emb_ds.weight.data
        self.emb_ds = nn.Embedding(newD, old_weight.size(1)).to(old_weight.device)
        self.emb_ds.weight.data[:self.D] = old_weight
        nn.init.zeros_(self.emb_ds.weight.data[self.D:])   # or small normal init
        self.emb_ds.weight.data[self.ref_id].zero_()
        self.emb_ds.weight.register_hook(self._zero_ref_hook)

        # extend ref mask
        new_mask = torch.ones(newD, 1, device=self.mask_ref.device)
        new_mask[:self.D] = self.mask_ref
        new_mask[self.ref_id] = 0.0
        self.register_buffer('mask_ref', new_mask)
        self.D = newD

    def regularizer(self, weight=1e-4):
        # small L2 on non-ref ds embeddings so shifts don't drift; joints too
        l2 = (self.emb_ds.weight.pow(2).sum() + self.emb_k.weight.pow(2).sum())
        return weight * l2

class MultiHandKeypoint(nn.Module):
    def __init__(self, single_hand_joint_num, pretrained_backbone=False):
        super(MultiHandKeypoint, self).__init__()

        self.backbone_net = BackboneHRNet(variant='w48', out_channels=512, pretrained=pretrained_backbone)

        K = single_hand_joint_num
        W_in = cfg.input_img_shape[1]
        H_in = cfg.input_img_shape[0]
        Hf   = cfg.output_hm_shape[1]
        Wf   = cfg.output_hm_shape[2]
        self.simcc_split_ratio = 2.0

        # Compute number of z bins from range & bin size
        self.Zbins = int(round((Z_MAX_MM - Z_MIN_MM) / Z_BIN_MM)) + 1

        self.head = RTMCCHeadLite(
            in_channels=512,
            out_channels=K*2,                         # <--- K, not K*2
            input_size=(W_in, H_in),
            in_featuremap_size=(Wf, Hf),
            simcc_split_ratio=self.simcc_split_ratio,
            gau_cfg=dict(
                hidden_dims=512, s=128, expansion_factor=2,
                dropout_rate=0., drop_path=0., act_fn='SiLU',
                use_rel_bias=False, pos_enc=False),
            z_bins=self.Zbins,                     # <--- enable z head
            z_detach=False                         # set True if you want zero influence on x/y
        )

        self.shift_layer = DatasetJointShiftEmb(3, K*2, ref_dataset_id=1, emb_dim=16)

        self.batch_ctr = 0

    def inference(self, inputs, targets=None, meta_info=None):
        self.batch_ctr += 1
        input_img = inputs['img']

        fused_feat, hr_feats = self.backbone_net(input_img)

        pred_x, pred_y, pred_z = self.head((fused_feat,))   # pred_z may be None

        ds_id = meta_info['dataset_id']
        dx, dy, dz = self.shift_layer(ds_id)                # (B,K) each
        pred_x = SimCCShift1D.shift_logits(pred_x, dx)
        pred_y = SimCCShift1D.shift_logits(pred_y, dy)
        pred_z = SimCCShift1D.shift_logits(pred_z, dz)

        kps_xy = decode_simcc(
            pred_x, pred_y,
            split_ratio=self.simcc_split_ratio,
            img_wh=(cfg.input_img_shape[1], cfg.input_img_shape[0]),
            use_argmax=False
        )

        out = {
            'pred_x': pred_x,
            'pred_y': pred_y,
            'kps_xy': kps_xy,
            'features': hr_feats,
            'fused': fused_feat,
        }

        z_rel_mm = decode_simcc_z_mm(pred_z, Z_MIN_MM, Z_BIN_MM)       # (B,K)
        out['z_rel_mm']            = z_rel_mm

        return out


    def forward(self, inputs, targets=None, meta_info=None):
        self.batch_ctr += 1
        input_img = inputs['img']

        fused_feat, hr_feats = self.backbone_net(input_img)
        pred_x, pred_y, pred_z = self.head((fused_feat,))

        ds_id = meta_info['dataset_id'].int()
        dx, dy, dz = self.shift_layer(ds_id)                # (B,K) each
        pred_x = SimCCShift1D.shift_logits(pred_x, dx)
        pred_y = SimCCShift1D.shift_logits(pred_y, dy)
        pred_z = SimCCShift1D.shift_logits(pred_z, dz)

        kps_xy = decode_simcc(
            pred_x, pred_y,
            split_ratio=self.simcc_split_ratio,
            img_wh=(cfg.input_img_shape[1], cfg.input_img_shape[0]),
            use_argmax=False
        )

        out = {
            'pred_x': pred_x,
            'pred_y': pred_y,
            'kps_xy': kps_xy,
            'features': hr_feats,
            'fused': fused_feat,
        }

        # ---- x/y targets (your existing pipeline)
        gt_x, gt_y, keypoint_weights = build_simcc_targets(
            targets["joint_coord_untransformed"],
            img_wh=(cfg.input_img_shape[1], cfg.input_img_shape[0]),
            split_ratio=self.simcc_split_ratio,
            sigma=2.5,
            keypoint_weights=targets.get('keypoint_weights', None)
        )
        loss_kpt = simcc_loss(pred_x, pred_y, gt_x, gt_y, keypoint_weights=keypoint_weights)

        # ---- z targets & loss (if pred_z exists and targets provided)
        if pred_z is not None and ('joint_coord_untransformed' in targets):
            # targets['rootrel_depth_mm']: (B, K) in mm, root-relative; root joint can be 0 mm
            gt_z, kp_w, _ = build_z_simcc_targets(
                targets["joint_coord_untransformed"][..., 2],
                z_min_mm=Z_MIN_MM,
                z_max_mm=Z_MAX_MM,
                bin_size_mm=Z_BIN_MM,
                sigma_bins=Z_SIGMA_B,
                keypoint_weights=targets.get('keypoint_weights', None)
            )
            # reuse KL loss on z
            log_pz = F.log_softmax(pred_z, dim=-1)
            loss_z = F.kl_div(log_pz, gt_z, reduction='none').mean(dim=-1)  # (B, K)
            if kp_w is not None:
                loss_z = loss_z * kp_w
            loss_z = loss_z.mean()

            out['loss_z'] = loss_z

        out['loss_xy']    = loss_kpt
        out['loss_shifter'] = self.shift_layer.regularizer(1e-4)
        out['simcc_x'] = gt_x
        out['simcc_y'] = gt_y

        intr = meta_info['intrinsics']             # {'focal': (fx,fy), 'princpt': (cx,cy)}
        focal   = torch.as_tensor(intr['focal'],   dtype=kps_xy.dtype, device=kps_xy.device)
        princpt = torch.as_tensor(intr['princpt'], dtype=kps_xy.dtype, device=kps_xy.device)

        z_rel_mm = decode_simcc_z_mm(pred_z, Z_MIN_MM, Z_BIN_MM)       # (B,K)
        Z_root_mm_right, Z_root_mm_left = meta_info['root_depth_mm_rl']["right"], meta_info['root_depth_mm_rl']["left"]
        Z_root_mm_right = Z_root_mm_right.to(kps_xy.device).to(kps_xy.dtype)  # (B,)
        Z_root_mm_left = Z_root_mm_left.to(kps_xy.device).to(kps_xy.dtype)  # (B,)

        kps_xy_trans = transform_keypoints_batched(kps_xy, meta_info["inv_trans"].to(kps_xy.device))

        kp_xyz_abs_mm = assemble_xyz(kps_xy_trans, z_rel_mm, Z_root_mm_right, Z_root_mm_left, focal, princpt)

        out['kp_xyz_abs_mm']       = kp_xyz_abs_mm 
        out['z_rel_mm']            = z_rel_mm

        return out
