"""
GTPJ framework model.

This file intentionally keeps only the gzsl-paper V1 path used by
``config/v1.yaml``:

- Progressive Semantic Enhancement (PSE)
- Frequency-Guided Visual Disentanglement (FGVD)
- Bidirectional Visual-Semantic Alignment (BVSA)
- Image-Conditioned Semantic Adapter (ICSA)
- Semantic-Guided Masked Prediction (SGMP)
- fixed add scoring: S_final = S_global + 0.2 * S_local
- CE, consistency, topology, BMDD, MPP, and negative semantic losses

Interface contract:
- forward(clip_features, is_train=False) returns a dict with ``clip_S_pp``.
- is_train=True  -> logits shape [B, n_seen].
- is_train=False -> logits shape [B, num_class].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_1d(length, sigma):
    x = torch.arange(-length // 2 + 1, length // 2 + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / torch.max(kernel)


def _autocast_disabled(tensor):
    return torch.amp.autocast(device_type=tensor.device.type, enabled=False)


def fgvd_select_patches(F_p, K=64):
    """Select top-K patches for Frequency-Guided Visual Disentanglement."""
    _, N, D = F_p.shape
    K = max(1, min(int(K), N))

    F_p_fp32 = F_p.float()
    x_freq = torch.fft.fft(F_p_fp32, dim=-1)
    sigma = D ** 0.5
    gs_k = _gaussian_kernel_1d(D, sigma).to(F_p_fp32.device)
    x_freq = torch.fft.fftshift(x_freq, dim=-1)
    x_freq = x_freq * gs_k
    x_freq = torch.fft.ifftshift(x_freq, dim=-1)
    x_lp = torch.fft.ifft(x_freq, dim=-1).real

    diff = F_p_fp32 / (torch.abs(x_lp - F_p_fp32) + 1e-6)
    patch_score = diff.abs().mean(dim=-1)
    _, topk_indices = torch.topk(patch_score, k=K, dim=1, largest=True)

    return topk_indices, patch_score


class ProgressiveSemanticSelfAttention(nn.Module):
    """Sentence-level self-attention inside Progressive Semantic Enhancement."""

    def __init__(self, dim, heads=1, dropout=0.5, inner_ratio=0.5):
        super().__init__()
        self.inner_ratio = float(inner_ratio)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(float(dropout))
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        attn_out = self.dropout(self.proj(attn_out))
        mixed = self.inner_ratio * attn_out + (1.0 - self.inner_ratio) * x
        return self.layer_norm(2.0 * mixed)


class BoxRelationalEmbedding(nn.Module):
    """Precomputed 24x24 patch-grid relational geometry embedding."""

    def __init__(self, grid_size=(24, 24), dim_g=64, wave_len=1000.0):
        super().__init__()
        self.grid_size = grid_size
        self.dim_g = dim_g
        self.wave_len = wave_len
        self.register_buffer("geometry_embedding", self._compute_embedding())

    def _compute_embedding(self):
        H, W = self.grid_size
        seq_len = H * W

        x = torch.arange(H).float()
        y = torch.arange(W).float()
        px_min = x.view(-1, 1).expand(-1, W).contiguous().view(-1)
        py_min = y.view(1, -1).expand(H, -1).contiguous().view(-1)
        px_max = px_min + 1
        py_max = py_min + 1

        cx = (px_min + px_max) * 0.5
        cy = (py_min + py_max) * 0.5
        w = px_max - px_min + 1.0
        h = py_max - py_min + 1.0

        delta_x = cx.unsqueeze(0) - cx.unsqueeze(1)
        delta_x = torch.clamp(torch.abs(delta_x / w.unsqueeze(0)), min=1e-3).log()
        delta_y = cy.unsqueeze(0) - cy.unsqueeze(1)
        delta_y = torch.clamp(torch.abs(delta_y / h.unsqueeze(0)), min=1e-3).log()
        delta_w = torch.log(w.unsqueeze(0) / w.unsqueeze(1))
        delta_h = torch.log(h.unsqueeze(0) / h.unsqueeze(1))
        pos_mat = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)

        feat_range = torch.arange(self.dim_g / 8).float()
        dim_mat = 1.0 / (self.wave_len ** (feat_range / (self.dim_g / 8)))
        dim_mat = dim_mat.view(1, 1, 1, -1)
        pos_mat = pos_mat.unsqueeze(-1) * 100.0
        mul_mat = (pos_mat * dim_mat).view(seq_len, seq_len, -1)
        embedding = torch.cat([mul_mat.sin(), mul_mat.cos()], dim=-1)
        return embedding.half()

class GeometryMultiHeadAttention(nn.Module):
    """Multi-head self-attention with TransZero-style geometry subtraction."""

    def __init__(self, dim_com, heads, dim_g=64, dropout=0.1):
        super().__init__()
        assert dim_com % heads == 0
        self.heads = heads
        self.d_k = dim_com // heads

        self.fc_q = nn.Linear(dim_com, dim_com)
        self.fc_k = nn.Linear(dim_com, dim_com)
        self.fc_v = nn.Linear(dim_com, dim_com)
        self.fc_o = nn.Linear(dim_com, dim_com)
        self.WGs = nn.ModuleList([nn.Linear(dim_g, 1, bias=True) for _ in range(heads)])
        self.ln = nn.LayerNorm(dim_com)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, geometry_emb):
        B, N, D = x.shape
        q = self.fc_q(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)
        v = self.fc_v(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)

        att = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        geo_flat = geometry_emb.float().reshape(-1, geometry_emb.shape[-1])
        geo_per_head = [
            layer(geo_flat).view(B, N, N, 1).permute(0, 3, 1, 2)
            for layer in self.WGs
        ]
        geo_weights = F.relu(torch.cat(geo_per_head, dim=1))
        att = F.softmax(att - geo_weights, dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return self.ln(x + self.fc_o(out))


class GeometryDecoupledEncoderLayer(nn.Module):
    """Geometry-decoupled visual encoder layer used by FGVD."""

    def __init__(self, dim_com, heads, dropout=0.1, dim_g=64):
        super().__init__()
        self.attn = GeometryMultiHeadAttention(dim_com, heads, dim_g, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim_com, dim_com * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim_com * 2, dim_com),
        )
        self.ln = nn.LayerNorm(dim_com)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, geometry_emb):
        x = self.attn(x, geometry_emb)
        return self.ln(x + self.dropout(self.ffn(x)))


class BidirectionalVisualSemanticAlignment(nn.Module):
    """Bidirectional Visual-Semantic Alignment (BVSA)."""

    def __init__(
        self,
        dim_f=768,
        dim_com=512,
        heads=4,
        dropout=0.1,
        weight_s2v=0.5,
        grid_size=(24, 24),
        dim_g=64,
    ):
        super().__init__()
        self.weight_s2v = weight_s2v
        self.embed_cv = nn.Linear(dim_f, dim_com)
        self.embed_text = nn.Linear(dim_f, dim_com)
        self.box_emb = BoxRelationalEmbedding(grid_size=grid_size, dim_g=dim_g)
        self.fgvd_encoder = GeometryDecoupledEncoderLayer(
            dim_com, heads, dropout, dim_g=dim_g
        )

        self.decoder_v2s = nn.TransformerDecoderLayer(
            d_model=dim_com,
            nhead=heads,
            dim_feedforward=dim_com * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder_s2v = nn.TransformerDecoderLayer(
            d_model=dim_com,
            nhead=heads,
            dim_feedforward=dim_com * 2,
            dropout=dropout,
            batch_first=True,
        )
    def geometry_for_indices(self, token_indices):
        K = token_indices.size(1)
        full = self.box_emb.geometry_embedding
        i_idx = token_indices.unsqueeze(-1).expand(-1, -1, K)
        j_idx = token_indices.unsqueeze(-2).expand(-1, K, -1)
        return full[i_idx, j_idx]

    def forward(
        self,
        patches,
        text,
        fgvd_select_k=0,
    ):
        B = patches.size(0)
        with _autocast_disabled(patches):
            topk_indices, _ = fgvd_select_patches(
                patches.float(),
                K=fgvd_select_k,
            )
        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, patches.size(-1))
        patches = torch.gather(patches, dim=1, index=idx_exp)

        vis = self.embed_cv(patches)
        geometry_emb = self.geometry_for_indices(topk_indices)
        memory = self.fgvd_encoder(vis, geometry_emb)

        txt_com = self.embed_text(text)
        if txt_com.dim() == 2:
            txt_batch = txt_com.unsqueeze(0).expand(B, -1, -1)
        elif txt_com.dim() == 3:
            if txt_com.size(0) != B:
                raise ValueError(
                    "Batched BVSA text must have shape [B, C, D] with the same B "
                    f"as patches; got text batch {txt_com.size(0)} and patches batch {B}."
                )
            txt_batch = txt_com
        else:
            raise ValueError(
                "BVSA text must be [C, D] for shared class text or [B, C, D] "
                f"for sample-conditioned class text; got {tuple(text.shape)}."
            )

        F_p_v2s = self.decoder_v2s(tgt=txt_batch, memory=memory)
        F_p_s2v = self.decoder_s2v(tgt=memory, memory=txt_batch)

        v2s_n = F.normalize(F_p_v2s, dim=-1)
        txt_batch_n = F.normalize(txt_batch, dim=-1)
        score_v2s = (v2s_n * txt_batch_n).sum(dim=-1)

        s2v_pooled = F_p_s2v.mean(dim=1)
        s2v_n = F.normalize(s2v_pooled, dim=-1)
        txt_single = F.normalize(txt_com, dim=-1)
        if txt_single.dim() == 2:
            score_s2v = s2v_n @ txt_single.T
        else:
            score_s2v = torch.einsum("bd,bcd->bc", s2v_n, txt_single)

        local_score = self.weight_s2v * score_s2v + (1.0 - self.weight_s2v) * score_v2s
        return {
            "local_score": local_score,
            "score_s2v": score_s2v,
            "score_v2s": score_v2s,
            "fgvd_selected_patches": patches,
            "fgvd_patch_z": vis,
            "fgvd_memory": memory,
        }


class GTPJ(nn.Module):
    """GTPJ framework with PSE, ICSA, FGVD, BVSA, and SGMP."""

    def __init__(
        self,
        config,
        seenclass,
        unseenclass,
        seen_text_embeds,
        unseen_text_embeds,
        seen_sentence_embeds=None,
    ):
        super().__init__()
        self.config = config
        self.nclass = int(config.num_class)
        self.dim_f = int(config.dim_f_clip)

        seen_ids = torch.as_tensor(seenclass, dtype=torch.long)
        unseen_ids = torch.as_tensor(unseenclass, dtype=torch.long)
        if seen_ids.dim() != 1 or unseen_ids.dim() != 1:
            raise ValueError("seenclass and unseenclass must be one-dimensional global ids.")
        if seen_ids.unique().numel() != seen_ids.numel():
            raise ValueError("seenclass contains duplicate global ids.")
        if unseen_ids.unique().numel() != unseen_ids.numel():
            raise ValueError("unseenclass contains duplicate global ids.")
        if torch.isin(seen_ids, unseen_ids).any():
            raise ValueError("seenclass and unseenclass must not overlap.")
        combined_ids = torch.cat([seen_ids, unseen_ids]).sort().values
        if not torch.equal(combined_ids, torch.arange(self.nclass, dtype=torch.long)):
            raise ValueError("seenclass and unseenclass must cover every global class exactly once.")
        if tuple(seen_text_embeds.shape) != (seen_ids.numel(), self.dim_f):
            raise ValueError("seen_text_embeds must have shape [C_seen, D].")
        if tuple(unseen_text_embeds.shape) != (unseen_ids.numel(), self.dim_f):
            raise ValueError("unseen_text_embeds must have shape [C_unseen, D].")

        self.register_buffer(
            "seenclass", seen_ids, persistent=False
        )
        self.register_buffer(
            "unseenclass", unseen_ids, persistent=False
        )

        self.seen_text_embeds = nn.Parameter(
            F.normalize(seen_text_embeds, dim=1), requires_grad=False
        )
        self.unseen_text_embeds = nn.Parameter(
            F.normalize(unseen_text_embeds, dim=1), requires_grad=False
        )

        fixed_route = {
            "use_pse_self_attention": True,
            "pse_apply_unseen": False,
            "use_fgvd_geometry": True,
            "fgvd_select_sigma": 0.0,
            "fgvd_select_largest": True,
            "fgvd_select_formula": "v2_abs_mean",
            "use_icsa": True,
            "bvsa_text_mode": "conditional",
            "use_sgmp": True,
            "sgmp_context_mode": "fgvd_main_memory",
            "sgmp_text_mode": "conditional",
            "consist_dynamic": True,
        }
        for name, expected in fixed_route.items():
            if hasattr(config, name) and getattr(config, name) != expected:
                raise ValueError(
                    f"V1 fixes {name}={expected!r}; "
                    f"got {getattr(config, name)!r}."
                )

        self.pse_outer_ratio = float(config.pse_outer_ratio)
        if seen_sentence_embeds is None:
            raise ValueError("V1 requires seen_sentence_embeds.")
        if (
            seen_sentence_embeds.dim() != 3
            or seen_sentence_embeds.size(0) != seen_ids.numel()
            or seen_sentence_embeds.size(-1) != self.dim_f
        ):
            raise ValueError("seen_sentence_embeds must have shape [C_seen, M, D].")
        self.seen_sentence_embeds = nn.Parameter(
            F.normalize(seen_sentence_embeds, dim=-1), requires_grad=False
        )
        self.pse_module = ProgressiveSemanticSelfAttention(
            dim=self.dim_f,
            heads=int(config.pse_heads),
            dropout=float(config.pse_dropout),
            inner_ratio=float(config.pse_inner_ratio),
        )

        tf_common_dim = int(config.tf_common_dim)
        tf_heads = int(config.tf_heads)
        tf_dropout = float(config.tf_dropout)
        weight_s2v = float(config.weight_s2v)
        if float(config.local_weight) != 0.2:
            raise ValueError(
                "V1 fixes local_weight=0.2; "
                f"got {config.local_weight!r}."
            )
        self.local_weight = 0.2
        if str(config.score_mode) != "add":
            raise ValueError("V1 requires score_mode='add'.")

        self.bvsa_module = BidirectionalVisualSemanticAlignment(
            dim_f=self.dim_f,
            dim_com=tf_common_dim,
            heads=tf_heads,
            dropout=tf_dropout,
            weight_s2v=weight_s2v,
            grid_size=(24, 24),
            dim_g=64,
        )

        self.sgmp_topk = int(config.sgmp_topk)
        self.sgmp_neg_margin = float(config.sgmp_neg_margin)
        sgmp_hidden = int(config.sgmp_hidden)
        self.sgmp_predictor = nn.Sequential(
            nn.Linear(tf_common_dim * 2, sgmp_hidden),
            nn.LayerNorm(sgmp_hidden),
            nn.GELU(),
            nn.Linear(sgmp_hidden, tf_common_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.fgvd_select_k = int(config.fgvd_select_k)
        if not 0 < self.fgvd_select_k < 576:
            raise ValueError("V1 requires 0 < fgvd_select_k < 576.")
        icsa_hidden = int(config.icsa_hidden)
        self.icsa_module = nn.Sequential(
            nn.Linear(self.dim_f, icsa_hidden),
            nn.LayerNorm(icsa_hidden),
            nn.GELU(),
            nn.Linear(icsa_hidden, self.dim_f),
        )
        with torch.no_grad():
            self.icsa_module[-1].weight.zero_()
            self.icsa_module[-1].bias.zero_()
        self.icsa_ratio = float(config.icsa_ratio)
        if self.icsa_ratio <= 0:
            raise ValueError("V1 requires icsa_ratio > 0.")

    def get_adapted_seen_text(self):
        sentence_embeds = self.seen_sentence_embeds
        base = sentence_embeds.mean(dim=1)
        attn = self.pse_module(sentence_embeds).mean(dim=1)
        ratio = self.pse_outer_ratio
        adapted = ratio * attn + (1.0 - ratio) * base
        return F.normalize(adapted, dim=1)

    def get_adapted_unseen_text(self):
        return self.unseen_text_embeds

    def _make_all_text(self, device, dtype):
        seen_text = self.get_adapted_seen_text().to(device=device, dtype=dtype)
        unseen_text = self.get_adapted_unseen_text().to(device=device, dtype=dtype)
        all_text = torch.zeros(self.nclass, self.dim_f, device=device, dtype=dtype)
        all_text[self.seenclass.to(device)] = seen_text
        all_text[self.unseenclass.to(device)] = unseen_text
        return all_text

    def _topology_pearson_loss(self, enh_text=None):
        if enh_text is None:
            adapted_seen = self.get_adapted_seen_text()
            device = adapted_seen.device
            dtype = adapted_seen.dtype
        else:
            device = enh_text.device
            dtype = enh_text.dtype

        base_text = torch.zeros(self.nclass, self.dim_f, device=device, dtype=dtype)
        seen_idx = self.seenclass.to(device)
        unseen_idx = self.unseenclass.to(device)
        base_text[seen_idx] = self.seen_text_embeds.to(device=device, dtype=dtype)
        base_text[unseen_idx] = self.unseen_text_embeds.to(device=device, dtype=dtype)

        if enh_text is None:
            enh_text = torch.zeros_like(base_text)
            enh_text[seen_idx] = adapted_seen.to(device=device, dtype=dtype)
            enh_text[unseen_idx] = base_text[unseen_idx]

        base_text = F.normalize(base_text.float(), dim=-1)
        enh_text = F.normalize(enh_text.float(), dim=-1)
        base_sim = base_text @ base_text.T
        if enh_text.dim() == 2:
            enh_sim = enh_text @ enh_text.T
        else:
            enh_sim = torch.matmul(enh_text, enh_text.transpose(-1, -2))

        off_diag = ~torch.eye(self.nclass, dtype=torch.bool, device=device)
        base_vec = base_sim.detach()[off_diag]
        if enh_sim.dim() == 2:
            enh_vec = enh_sim[off_diag].unsqueeze(0)
        else:
            enh_vec = enh_sim[:, off_diag]
        base_vec = base_vec.unsqueeze(0).expand_as(enh_vec)

        enh_centered = enh_vec - enh_vec.mean(dim=1, keepdim=True)
        base_centered = base_vec - base_vec.mean(dim=1, keepdim=True)
        numerator = (enh_centered * base_centered).sum(dim=1)
        denominator = (
            torch.sqrt((enh_centered ** 2).sum(dim=1) + 1e-8)
            * torch.sqrt((base_centered ** 2).sum(dim=1) + 1e-8)
        )
        return (1.0 - numerator / denominator).mean()

    def _semantic_guided_masked_prediction_loss(
        self,
        labels,
        selected_patches=None,
        selected_patch_z=None,
        selected_memory=None,
        all_text_cond=None,
    ):
        device = labels.device
        if selected_patches is None or selected_patch_z is None:
            raise ValueError("SGMP requires selected patches and FGVD patch_z.")
        if selected_memory is None:
            raise ValueError("SGMP requires the main-path FGVD memory.")
        if all_text_cond is None:
            raise ValueError("SGMP requires image-conditioned class text.")
        sgmp_patches = selected_patches

        B, N, _ = sgmp_patches.shape
        if N < 2:
            zero = torch.tensor(0.0, device=device)
            return zero, zero
        k = max(1, min(int(self.sgmp_topk), N - 1))
        labels = labels.to(device=device, dtype=torch.long)
        batch_idx = torch.arange(B, device=device)
        class_text = all_text_cond[batch_idx, labels].to(
            device=device, dtype=sgmp_patches.dtype
        )

        with torch.no_grad():
            patch_n = F.normalize(sgmp_patches.float(), dim=-1)
            text_n = F.normalize(class_text.float(), dim=-1)
            patch_score = torch.einsum("bnd,bd->bn", patch_n, text_n)
            _, masked_idx = torch.topk(patch_score, k=k, dim=1, largest=True)

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_idx, True)
        keep = ~mask

        patch_z = selected_patch_z
        keep_f = keep.unsqueeze(-1).to(selected_memory.dtype)
        context = (selected_memory * keep_f).sum(dim=1) / keep_f.sum(dim=1).clamp_min(1.0)

        target = patch_z[mask].view(B, k, -1).mean(dim=1).detach()

        text_z = self.bvsa_module.embed_text(class_text)
        pred = self.sgmp_predictor(torch.cat([context, text_z], dim=-1))
        pos_sim = F.cosine_similarity(pred, target, dim=-1)
        loss_mpp = (1.0 - pos_sim).mean()

        seen = self.seenclass.to(device)
        label_map = torch.full((self.nclass,), -1, device=device, dtype=torch.long)
        label_map[seen] = torch.arange(seen.numel(), device=device)
        local_labels = label_map[labels]
        if (local_labels < 0).any():
            raise ValueError("SGMP expects global labels from seen classes.")
        neg_local = (local_labels + 1) % seen.numel()
        neg_labels = seen[neg_local]
        neg_text = all_text_cond[batch_idx, neg_labels].to(
            device=device, dtype=sgmp_patches.dtype
        )

        neg_text_z = self.bvsa_module.embed_text(neg_text)
        pred_neg = self.sgmp_predictor(torch.cat([context.detach(), neg_text_z], dim=-1))
        neg_sim = F.cosine_similarity(pred_neg, target, dim=-1)
        loss_neg = F.relu(neg_sim - pos_sim.detach() + self.sgmp_neg_margin).mean()
        return loss_mpp, loss_neg

    def forward(self, clip_features, is_train=False):
        if clip_features.dim() != 3 or clip_features.size(1) != 577:
            raise ValueError(
                "V1 requires clip_features with shape [B, 577, D]; "
                f"got {tuple(clip_features.shape)}."
            )
        if clip_features.size(2) != self.dim_f:
            raise ValueError(
                f"V1 requires feature dimension D={self.dim_f}; "
                f"got D={clip_features.size(2)}."
            )
        cls_token = clip_features[:, 0, :]
        patches = clip_features[:, 1:, :]

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        all_text = self._make_all_text(patches.device, patches.dtype)
        vis_n = F.normalize(cls_token, dim=1)
        pi_x = F.normalize(self.icsa_module(cls_token), dim=-1)
        all_text_cond = all_text.unsqueeze(0).expand(cls_token.size(0), -1, -1).clone()
        seen_idx = self.seenclass.to(patches.device)
        all_text_cond[:, seen_idx, :] = (
            all_text[seen_idx].unsqueeze(0)
            + self.icsa_ratio * pi_x.unsqueeze(1)
        )
        text_n_cond = F.normalize(all_text_cond, dim=-1)
        global_logits = (vis_n.unsqueeze(1) * text_n_cond).sum(dim=-1) * logit_scale

        bvsa_out = self.bvsa_module(
            patches,
            all_text_cond,
            fgvd_select_k=self.fgvd_select_k,
        )
        local_logits = bvsa_out["local_score"]
        final_logits = global_logits + 0.2 * local_logits

        if is_train:
            logits = final_logits[:, self.seenclass.to(final_logits.device)]
        else:
            logits = final_logits

        return {
            "logits": logits,
            "final_logits": final_logits,
            "global_logits": global_logits,
            "local_logits": local_logits,
            "clip_S_pp": logits,
            "score_s2v": bvsa_out["score_s2v"],
            "score_v2s": bvsa_out["score_v2s"],
            "sgmp_selected_patches": bvsa_out["fgvd_selected_patches"],
            "sgmp_patch_z": bvsa_out["fgvd_patch_z"],
            "sgmp_memory": bvsa_out["fgvd_memory"],
            "all_text_cond": all_text_cond,
        }

    def _global_to_seen_labels(self, labels):
        labels = labels.to(device=self.seenclass.device, dtype=torch.long)
        label_map = torch.full(
            (self.nclass,), -1, device=self.seenclass.device, dtype=torch.long
        )
        label_map[self.seenclass] = torch.arange(
            self.seenclass.numel(), device=self.seenclass.device
        )
        seen_labels = label_map[labels]
        if (seen_labels < 0).any():
            raise ValueError("Training labels must be global ids from seen classes.")
        return seen_labels

    def compute_loss(self, in_package):
        logits = in_package["logits"]
        labels = in_package["batch_label"]
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)
        labels = labels.to(device=logits.device, dtype=torch.long)
        seen_labels = self._global_to_seen_labels(labels).to(logits.device)

        loss_ce = F.cross_entropy(logits, seen_labels)
        loss = loss_ce

        global_logits = in_package.get("global_logits")
        local_logits = in_package.get("local_logits")
        loss_consist = torch.tensor(0.0, device=logits.device)
        lambda_consist = float(self.config.lambda_consist)
        if global_logits is not None and local_logits is not None and lambda_consist > 0:
            temperature = float(self.config.consist_temp)
            seen_idx = self.seenclass.to(logits.device)
            global_seen = global_logits[:, seen_idx].detach()
            local_seen = local_logits[:, seen_idx]
            global_probability = F.softmax(global_seen / temperature, dim=-1)
            local_log_probability = F.log_softmax(local_seen / temperature, dim=-1)
            loss_consist = F.kl_div(
                local_log_probability, global_probability, reduction="batchmean"
            ) * (temperature * temperature)
            gamma = float(self.config.consist_dynamic_gamma)
            with torch.no_grad():
                scale = 1.0 / (1.0 + gamma * loss_consist.detach())
            loss = loss + (lambda_consist * scale) * loss_consist

        loss_topo = torch.tensor(0.0, device=logits.device)
        lambda_topo = float(self.config.lambda_topo_pearson)
        if lambda_topo > 0:
            loss_topo = self._topology_pearson_loss()
            loss = loss + lambda_topo * loss_topo

        loss_mpp = torch.tensor(0.0, device=logits.device)
        loss_neg = torch.tensor(0.0, device=logits.device)
        lambda_mpp = float(self.config.lambda_mpp)
        lambda_neg = float(self.config.lambda_neg)
        if lambda_mpp > 0 or lambda_neg > 0:
            loss_mpp, loss_neg = self._semantic_guided_masked_prediction_loss(
                labels,
                selected_patches=in_package.get("sgmp_selected_patches"),
                selected_patch_z=in_package.get("sgmp_patch_z"),
                selected_memory=in_package.get("sgmp_memory"),
                all_text_cond=in_package.get("all_text_cond"),
            )
            loss = loss + lambda_mpp * loss_mpp
            loss = loss + lambda_neg * loss_neg

        score_s2v = in_package.get("score_s2v")
        score_v2s = in_package.get("score_v2s")
        loss_bmdd = torch.tensor(0.0, device=logits.device)
        lambda_bmdd = float(self.config.lambda_bmdd)
        if lambda_bmdd > 0 and score_s2v is not None and score_v2s is not None:
            temperature = float(self.config.msdn_temp)
            seen_idx = self.seenclass.to(logits.device)
            s2v_seen = score_s2v[:, seen_idx] / temperature
            v2s_seen = score_v2s[:, seen_idx] / temperature
            p_s2v = F.softmax(s2v_seen, dim=-1)
            p_v2s = F.softmax(v2s_seen, dim=-1)
            log_p_s2v = F.log_softmax(s2v_seen, dim=-1)
            log_p_v2s = F.log_softmax(v2s_seen, dim=-1)
            kl_s2v_to_v2s = F.kl_div(log_p_v2s, p_s2v.detach(), reduction="batchmean")
            kl_v2s_to_s2v = F.kl_div(log_p_s2v, p_v2s.detach(), reduction="batchmean")
            loss_bmdd = (temperature * temperature / 2.0) * (
                kl_s2v_to_v2s + kl_v2s_to_s2v
            )
            loss = loss + lambda_bmdd * loss_bmdd

        return {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_consist": loss_consist,
            "loss_topo": loss_topo,
            "loss_bmdd": loss_bmdd,
            "loss_mpp": loss_mpp,
            "loss_neg": loss_neg,
        }
