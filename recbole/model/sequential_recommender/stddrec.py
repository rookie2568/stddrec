"""STDDRec: Stable-Transient Dual-Stream Diffusion for Sequential Recommendation.

Implemented following the paper (Sec. 4):
  4.1 embedding layer                     -> item + learnable positional embeddings
  4.2.1 past / recent encoders            -> self-attention  /  tGRU (conv1d + GRU)
  4.2.2 stable-transient disentanglement  -> interest prototypes + consistency gate + residual
  4.3.1 interest shift / behavioral ambiguity
  4.3.2 adaptive dual-stream noise schedules
  4.3.3 dual-stream forward process
  4.3.4 shared MoE denoising network
  4.3.5 reverse process + uncertainty from expert disagreement
  4.4   uncertainty-guided fusion
  4.5   max-pooling prediction layer
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

def masked_mean(x, mask):
    """x: [B, L, d], mask: [B, L] in {0,1}."""
    m = mask.unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom

class Block(nn.Module):
    """Pre-LN Transformer block."""

    def __init__(self, d, num_heads, dropout, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * ff_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * ff_mult, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x

class MLP(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def forward(self, x):
        return x + self.net(x)

class MoEDenoiser(nn.Module):
    def __init__(self, d, num_experts, num_heads, num_layers, dropout, expert_hidden,
                 use_moe=True):
        super().__init__()
        self.d = d
        self.num_experts = num_experts
        self.use_moe = use_moe

        self.time_mlp = nn.Sequential(
            nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d)
        )
        self.branch_emb = nn.Embedding(2, d)
        self.inp = nn.Linear(3 * d + 2, d)
        self.blocks = nn.ModuleList(
            [Block(d, num_heads, dropout) for _ in range(num_layers)]
        )

        self.shared_expert = MLP(d, expert_hidden, dropout)
        n_res = num_experts if use_moe else 1
        self.residual_experts = nn.ModuleList(
            [MLP(d, expert_hidden, dropout) for _ in range(n_res)]
        )
        self.router = nn.Sequential(
            nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, n_res)
        )

    def forward(self, z, t, branch_id, shift, ambiguity):
        """z: [B,K,d]; t: [B]; branch_id: [B]; shift/ambiguity: [B]."""
        b, k, d = z.shape
        e_t = self.time_mlp(timestep_embedding(t, d))
        e_m = self.branch_emb(branch_id)
        cond = torch.cat([e_t, e_m, shift.unsqueeze(-1), ambiguity.unsqueeze(-1)], dim=-1)

        q = torch.cat([z, cond.unsqueeze(1).expand(-1, k, -1)], dim=-1)
        h = self.inp(q)
        for blk in self.blocks:
            h = blk(h)

        shared = self.shared_expert(h)
        cand = torch.stack([shared + e(h) for e in self.residual_experts], dim=1)

        pooled = h.mean(dim=1)
        logits = self.router(torch.cat([pooled, shift.unsqueeze(-1), ambiguity.unsqueeze(-1)], -1))
        routing = torch.softmax(logits, dim=-1)

        eps_hat = torch.einsum("bn,bnkd->bkd", routing, cand)
        return eps_hat, cand, routing

class STDDRec(nn.Module):
    def __init__(self, args, item_size):
        super().__init__()
        self.args = args
        d = args.hidden_size
        L = args.max_seq_length
        self.d = d
        self.item_size = item_size
        self.num_interests = args.num_interests
        self.last_n = args.uncertainty_last_steps

        self.item_emb = nn.Embedding(item_size, d, padding_idx=0)
        self.pos_past = nn.Embedding(L, d)
        self.pos_rec = nn.Embedding(L, d)
        self.emb_ln = nn.LayerNorm(d)
        self.emb_drop = nn.Dropout(args.dropout)

        self.past_layers = nn.ModuleList(
            [Block(d, args.num_heads, args.dropout) for _ in range(args.num_past_layers)]
        )
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.gru = nn.GRU(d, d, num_layers=1, batch_first=True)
        self.ln_recent = nn.LayerNorm(d)

        self.prototypes = nn.Parameter(torch.randn(args.num_interests, d) * 0.02)
        self.ln_zp = nn.LayerNorm(d)
        self.ln_zr = nn.LayerNorm(d)
        self.gate = nn.Linear(2 * d, d)

        self.denoiser = MoEDenoiser(
            d, args.num_experts, args.num_heads, args.num_denoise_layers,
            args.dropout, args.expert_hidden, use_moe=not args.no_moe,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _fix_pad_mask(kpm):
        all_masked = kpm.all(dim=1)
        if all_masked.any():
            kpm = kpm.clone()
            kpm[all_masked, 0] = False
        return kpm

    def _embed(self, ids, pos):
        x = self.item_emb(ids) + pos.weight.unsqueeze(0)[:, : ids.size(1)]
        return self.emb_drop(self.emb_ln(x))

    def encode_past(self, ids, mask):
        x = self._embed(ids, self.pos_past)
        kpm = self._fix_pad_mask(mask < 0.5)
        for blk in self.past_layers:
            x = blk(x, kpm)
        return x

    def encode_recent(self, ids, mask):
        x = self._embed(ids, self.pos_rec)
        h = F.gelu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        h, _ = self.gru(h)
        return self.ln_recent(h)

    def _assign(self, h, mask):
        """Eq.(10)-(11): prototype assignment and aggregation to interest space."""
        logits = torch.einsum("bld,kd->blk", h, self.prototypes)
        a = torch.softmax(logits, dim=-1) * mask.unsqueeze(-1)
        z = torch.einsum("blk,bld->bkd", a, h)
        if self.args.assign_norm == "mean":
            z = z / mask.sum(dim=1).clamp_min(1.0).view(-1, 1, 1)
        return z

    def _ambiguity(self, ids, mask):
        hs = []
        for _ in range(2):
            keep = (torch.rand_like(mask) > self.args.ambiguity_mask_prob).float()
            keep = keep * (mask > 0).float()
            need = keep.sum(dim=1) < 1
            if need.any():
                keep[need, 0] = mask[need, 0]
            h = self.encode_recent((ids * keep.long()), mask * keep)
            hs.append(masked_mean(h, mask * keep))
        return (1.0 - F.cosine_similarity(hs[0], hs[1], dim=-1)) / 2.0

    def _schedules(self, shift, ambiguity, T):
        dev = shift.device
        base = torch.linspace(self.args.beta_min, self.args.beta_max, T, device=dev)

        if self.args.no_ads:
            rho = torch.ones_like(shift)
        else:
            rho = (1.0 + self.args.gamma_A * ambiguity
                   - self.args.gamma_D * shift * (1.0 - ambiguity))
            rho = rho.clamp(self.args.rho_min, self.args.rho_max)

        beta_s = (self.args.kappa_s * base).clamp(self.args.beta_min, self.args.beta_max)
        beta_tr = (rho.unsqueeze(1) * base.unsqueeze(0)).clamp(self.args.beta_min, self.args.beta_max)

        one = torch.ones(shift.size(0), 1, device=dev)
        abar_s = torch.cat([one, torch.cumprod(1.0 - beta_s, dim=0).unsqueeze(0).expand(shift.size(0), -1)], dim=1)
        abar_tr = torch.cat([one, torch.cumprod(1.0 - beta_tr, dim=1)], dim=1)
        return abar_s, abar_tr

    def _denoise(self, z, t, branch_id, shift, ambiguity):
        b = z.size(0)
        if torch.is_tensor(t):
            t = t.to(z.device).long()
        else:
            t = torch.full((b,), int(t), dtype=torch.long, device=z.device)
        return self.denoiser(z, t, branch_id, shift, ambiguity)

    def _diffusion_step_losses(self, z0, abar, branch_id, shift, ambiguity, T):
        """Single randomly-sampled step: L_diff, calibration pair (U, e) and z0_hat."""
        b = z0.size(0)
        t = torch.randint(1, T + 1, (b,), device=z0.device)
        a = abar.gather(1, t.unsqueeze(1)).view(b, 1, 1)
        eps = torch.randn_like(z0)
        z_t = a.sqrt() * z0 + (1.0 - a).sqrt() * eps

        eps_hat, cand, routing = self._denoise(z_t, t, branch_id, shift, ambiguity)

        diff = ((eps - eps_hat) ** 2).mean()
        e = ((eps - eps_hat) ** 2).mean(dim=(1, 2))
        u = (((cand - eps_hat.unsqueeze(1)) ** 2) * routing[:, :, None, None]).sum(dim=1).mean(dim=(1, 2))
        z0_hat = (z_t - (1.0 - a).sqrt() * eps_hat) / a.sqrt()
        return diff, u, e, routing, z0_hat

    def reverse(self, z0, abar, branch_id, shift, ambiguity):
        """Eq.(38)-(39): deterministic DDIM reverse refinement."""
        b, k, d = z0.shape
        T = abar.size(1) - 1
        eps_T = torch.randn_like(z0)
        aT = abar[:, T].view(b, 1, 1)
        z = aT.sqrt() * z0 + (1.0 - aT).sqrt() * eps_T

        u_acc = torch.zeros(b, device=z0.device)
        n_u = 0
        z0_hat = z0
        for t in range(T, 0, -1):
            eps_hat, cand, routing = self._denoise(z, t, branch_id, shift, ambiguity)
            a_t = abar[:, t].view(b, 1, 1)
            a_p = abar[:, t - 1].view(b, 1, 1)

            z0_hat = (z - (1.0 - a_t).sqrt() * eps_hat) / a_t.sqrt()
            if t <= self.last_n:
                u_t = (((cand - eps_hat.unsqueeze(1)) ** 2) * routing[:, :, None, None]).sum(dim=1).mean(dim=(1, 2))
                u_acc = u_acc + u_t
                n_u += 1
            z = a_p.sqrt() * z0_hat + (1.0 - a_p).sqrt() * eps_hat
        return z0_hat, u_acc / max(n_u, 1), routing

    def predict(self, z_fusion):
        """z_fusion: [B,K,d] -> scores over items 1..M, via max-pool over K."""
        emb = self.item_emb.weight[1:]
        b, k, _ = z_fusion.shape
        outs = []
        step = 4096
        for i in range(0, emb.size(0), step):
            chunk = emb[i:i + step]
            s = torch.einsum("bkd,md->bkm", z_fusion, chunk)
            outs.append(s.max(dim=1).values)
        return torch.cat(outs, dim=1)

    def forward(self, past_ids, past_mask, rec_ids, rec_mask, targets=None):
        args = self.args
        T = args.diffusion_steps
        b = past_ids.size(0)
        dev = past_ids.device

        h_p = self.encode_past(past_ids, past_mask)
        h_r = self.encode_recent(rec_ids, rec_mask)

        z_p = self.ln_zp(self._assign(h_p, past_mask))
        z_r = self.ln_zr(self._assign(h_r, rec_mask))

        if args.no_st:
            z_s, z_tr = z_r, z_r
        else:
            g = torch.sigmoid(self.gate(torch.cat([z_p, z_r], dim=-1)))
            z_s = g * z_p + (1.0 - g) * z_r
            z_tr = z_r - z_s

        shift = (1.0 - F.cosine_similarity(z_p.mean(1), z_r.mean(1), dim=-1)) / 2.0
        ambiguity = self._ambiguity(rec_ids, rec_mask)

        abar_s, abar_tr = self._schedules(shift, ambiguity, T)
        branch_s = torch.zeros(b, dtype=torch.long, device=dev)
        branch_tr = torch.ones(b, dtype=torch.long, device=dev)

        losses = {}
        z_s_hat = z_tr_hat = None

        if self.training:
            d_s, u_s, e_s, r_s, zs_1 = self._diffusion_step_losses(
                z_s, abar_s, branch_s, shift, ambiguity, T)
            d_t, u_t, e_t, r_t, ztr_1 = self._diffusion_step_losses(
                z_tr, abar_tr, branch_tr, shift, ambiguity, T)
            losses["diff"] = 0.5 * (d_s + d_t)
            if not args.no_cal:
                losses["cal"] = ((u_s - e_s) ** 2).mean() + ((u_t - e_t) ** 2).mean()
            if not args.no_bal:
                ne = r_s.size(1)
                rb = torch.cat([r_s.mean(0), r_t.mean(0)], dim=0).view(2, ne).mean(0)
                losses["bal"] = ((rb - 1.0 / ne) ** 2).sum()
            if args.train_refine == "single" and not args.no_reverse:
                z_s_hat, u_s_bar, z_tr_hat, u_tr_bar = zs_1, u_s, ztr_1, u_t

        if z_s_hat is None:
            if args.no_reverse:
                z_s_hat, u_s_bar = z_s, torch.zeros(b, device=dev)
                z_tr_hat, u_tr_bar = z_tr, torch.zeros(b, device=dev)
            else:
                z_s_hat, u_s_bar, _ = self.reverse(z_s, abar_s, branch_s, shift, ambiguity)
                z_tr_hat, u_tr_bar, _ = self.reverse(z_tr, abar_tr, branch_tr, shift, ambiguity)

        if args.no_ugf:
            w_s = torch.full_like(u_s_bar, 0.5)
            w_tr = torch.full_like(u_tr_bar, 0.5)
        else:
            c_s = torch.exp(-u_s_bar)
            c_tr = torch.exp(-u_tr_bar)
            w_s = c_s / (c_s + c_tr + 1e-8)
            w_tr = 1.0 - w_s
        if args.fusion == "paper":
            z_fusion = w_s.view(b, 1, 1) * z_s_hat + w_tr.view(b, 1, 1) * z_tr_hat
        else:
            z_fusion = z_s_hat + w_tr.view(b, 1, 1) * z_tr_hat

        logits = self.predict(z_fusion)

        rec_loss = None
        if targets is not None:
            if args.loss == "bce":
                label = F.one_hot(targets - 1, num_classes=logits.size(1)).float()
                rec_loss = F.binary_cross_entropy_with_logits(logits, label)
            else:
                rec_loss = F.cross_entropy(logits, targets - 1)

        return {"logits": logits, "rec_loss": rec_loss, "losses": losses,
                "z_fusion": z_fusion, "w_s": w_s, "shift": shift, "ambiguity": ambiguity}
