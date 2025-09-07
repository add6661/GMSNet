
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from modules.dataset.megadepth import megadepth_warper
from modules.training import utils
from third_party.alike_wrapper import extract_alike_kpts



def loco_ranking_loss(
    feats: torch.Tensor,
    pos_pairs: torch.LongTensor,
    neg_pairs: torch.LongTensor,
    tau: float = 1e-2,
    delta: float = 7.6e-2,
    reduction: str = "mean",
) -> torch.Tensor:

    if feats.dim() != 2:
        raise ValueError("`feats` 应为 2-D (N, D) 张量")
    device = feats.device
    P = pos_pairs.size(0)

    sim_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _sim(a: int, b: int) -> torch.Tensor:
        key = (a, b) if a <= b else (b, a)
        if key not in sim_cache:
            sim_cache[key] = torch.dot(feats[a], feats[b])
        return sim_cache[key]

    def _group_pairs(pairs: torch.LongTensor) -> Dict[int, List[int]]:
        mapping: Dict[int, List[int]] = {}
        for a, b in pairs.tolist():
            mapping.setdefault(a, []).append(b)
            mapping.setdefault(b, []).append(a)
        return mapping

    pos_by_anchor = _group_pairs(pos_pairs)
    neg_by_anchor = _group_pairs(neg_pairs)

    losses = []
    for idx in range(P):
        i, j = pos_pairs[idx]
        s_alpha = _sim(i, j)
        anchor = i

        pos_cands = [p for p in pos_by_anchor.get(anchor, []) if p != j]
        neg_cands = neg_by_anchor.get(anchor, [])

        pos_keep = [p for p in pos_cands
                    if abs(_sim(anchor, p) - s_alpha) <= delta]
        neg_keep = [n for n in neg_cands
                    if abs(_sim(anchor, n) - s_alpha) <= delta]

        if len(pos_keep) == 0 and len(neg_keep) == 0:

            continue

        num = torch.tensor(1.0, device=device)
        for p in pos_keep:
            num = num + torch.sigmoid((_sim(anchor, p) - s_alpha) / tau)

        den = num.clone()
        for n in neg_keep:
            den = den + torch.sigmoid((_sim(anchor, n) - s_alpha) / tau)

        losses.append(-(torch.log(num) - torch.log(den)))

    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = torch.stack(losses)
    if reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        return loss.mean()



def dual_softmax_loss(X, Y, temp: float = 0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp(conf_matrix12).max(dim=-1)[0]
        conf21 = torch.exp(conf_matrix21).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device=X.device)
    loss = F.nll_loss(conf_matrix12, target) + F.nll_loss(conf_matrix21, target)
    return loss, conf


def smooth_l1_loss(input, target, beta: float = 2.0, size_average: bool = True):
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean() if size_average else loss.sum()


def fine_loss(f1, f2, pts1, pts2, fine_module, ws: int = 7):
    C, H, W = f1.shape
    N = len(pts1)


    with torch.no_grad():
        a = -(ws // 2)
        b = (ws // 2)
        offset_gt = (a - b) * torch.rand(N, 2, device=f1.device) + b
        pts2_random = pts2 + offset_gt

    patches1 = utils.crop_patches(
        f1.unsqueeze(0), (pts1 + 0.5).long(), size=ws
    ).view(C, N, ws * ws).permute(1, 2, 0)
    patches2 = utils.crop_patches(
        f2.unsqueeze(0), (pts2_random + 0.5).long(), size=ws
    ).view(C, N, ws * ws).permute(1, 2, 0)

    patches1, patches2 = fine_module(patches1, patches2)

    features = patches1.view(N, ws, ws, C)[:, ws // 2, ws // 2, :].view(N, 1, 1, C)
    patches2 = patches2.view(N, ws, ws, C)

    heatmap_match = (features * patches2).sum(-1)
    offset_coords = utils.subpix_softmax2d(heatmap_match)

    offset_gt = -offset_gt

    error = ((offset_coords - offset_gt) ** 2).sum(-1).mean()
    return error


def alike_distill_loss(kpts, img):
    C, H, W = kpts.shape
    kpts = kpts.permute(1, 2, 0)
    img = img.permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy() * 255

    with torch.no_grad():
        alike_kpts = torch.tensor(extract_alike_kpts(img), device=kpts.device)
        labels = torch.ones((H, W), dtype=torch.long, device=kpts.device) * 64  # 默认非关键点 bin=64
        offsets = (((alike_kpts / 8) - (alike_kpts / 8).long()) * 8).long()
        offsets = offsets[:, 0] + 8 * offsets[:, 1]
        labels[(alike_kpts[:, 1] / 8).long(), (alike_kpts[:, 0] / 8).long()] = offsets

    kpts = kpts.view(-1, C)
    labels = labels.view(-1)

    mask = labels < 64
    idxs_pos = mask.nonzero().flatten()
    idxs_neg = (~mask).nonzero().flatten()
    perm = torch.randperm(idxs_neg.size(0))[: len(idxs_pos) // 32]
    idxs = torch.cat([idxs_pos, idxs_neg[perm]])

    kpts = kpts[idxs]
    labels = labels[idxs]

    with torch.no_grad():
        predicted = kpts.max(dim=-1)[1]
        acc = (labels == predicted).float().mean()

    kpts = F.log_softmax(kpts, dim=-1)
    loss = F.nll_loss(kpts, labels, reduction="mean")
    return loss, acc


def keypoint_position_loss(kpts1, kpts2, pts1, pts2, softmax_temp: float = 1.0):

    C, H, W = kpts1.shape
    kpts1 = kpts1.permute(1, 2, 0) * softmax_temp
    kpts2 = kpts2.permute(1, 2, 0) * softmax_temp

    with torch.no_grad():
        x, y = torch.meshgrid(
            torch.arange(W, device=kpts1.device),
            torch.arange(H, device=kpts1.device),
            indexing="xy",
        )
        xy = torch.stack([x, y], dim=-1) * 8
        hashmap = torch.full((H * 8, W * 8, 2), -1, dtype=torch.long, device=kpts1.device)
        hashmap[pts1[:, 1].long(), pts1[:, 0].long()] = pts2.long()

        _, k1_off = kpts1.max(dim=-1)
        k1_off = torch.stack([k1_off % 8, k1_off // 8], dim=-1)
        coords1 = (xy + k1_off).view(-1, 2)

        gt = hashmap[coords1[:, 1], coords1[:, 0]]
        mask_valid = (gt >= 0).all(dim=-1)
        gt = gt[mask_valid]

        labels2 = (((gt / 8) - (gt / 8).long()) * 8).long()
        labels2 = labels2[:, 0] + 8 * labels2[:, 1]

    kpts2_sel = kpts2[(gt[:, 1] / 8).long(), (gt[:, 0] / 8).long()]
    k1_sel = F.log_softmax(kpts1.view(-1, C)[mask_valid], dim=-1)
    k2_sel = F.log_softmax(kpts2_sel, dim=-1)

    with torch.no_grad():
        _, labels1 = k1_sel.max(dim=-1)

    acc = (labels2 == k2_sel.max(dim=-1)[1]).float().mean()
    loss = F.nll_loss(k1_sel, labels1, reduction="mean") + F.nll_loss(k2_sel, labels2, reduction="mean")
    return loss, acc


def coordinate_classification_loss(coords1, pts1, pts2, conf):
    with torch.no_grad():
        coords1_det = pts1 * 8
        offsets1 = (((coords1_det / 8) - (coords1_det / 8).long()) * 8).long()
        labels1 = offsets1[:, 0] + 8 * offsets1[:, 1]

    coords1_log = F.log_softmax(coords1, dim=-1)
    acc = ((labels1 == coords1.max(dim=-1)[1]) & (conf > 0.1)).float().mean()

    loss = F.nll_loss(coords1_log, labels1, reduction="none")
    conf = conf / conf.sum()
    loss = (loss * conf).sum()
    return loss * 2.0, acc


def keypoint_loss(heatmap, target):
    return F.l1_loss(heatmap, target) * 3.0


def hard_triplet_loss(X, Y, margin: float = 0.5):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = torch.cdist(X, Y, p=2.0)
    dist_pos = torch.diag(dist_mat)
    dist_neg = dist_mat + 100.0 * torch.eye(
        *dist_mat.size(),
        dtype=dist_mat.dtype,
        device=(dist_mat.device if dist_mat.is_cuda else torch.device("cpu")),
    )
    dist_neg = dist_neg + dist_neg.le(0.01).float() * 100.0
    hard_neg = dist_neg.min(dim=1)[0]
    loss = torch.clamp(margin + dist_pos - hard_neg, min=0.0)
    return loss.mean()

