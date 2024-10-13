import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction="none"):
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (
        inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i)
                          for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i)
                          for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()

class InstMatcher(nn.Module):

    def __init__(self):
        super().__init__()
        self.alpha = 0.8
        self.beta = 0.2
        self.mask_score = dice_score

    def forward(self, outputs, targets):
        with torch.no_grad():
            B, N, _ = outputs["pred_masks"].shape
            pred_masks = outputs['pred_masks']
            pred_logits = outputs['pred_logits'].sigmoid()

            tgt_ids = []
            tgt_masks = []
            for batch_idx in range(targets.shape[0]):
                target = targets[batch_idx]
                for lbl in sorted(target.unique()):
                    if lbl==14:  # 14 is background
                        continue
                    tgt_ids.append(lbl.unsqueeze(0))
                    tgt_mask = torch.where(target==lbl, 1, 0)
                    tgt_masks.append(tgt_mask.unsqueeze(0))
            tgt_ids = torch.cat(tgt_ids)
            tgt_masks = torch.cat(tgt_masks).to(pred_masks) #.to(xxx) 类型转换

            pred_masks = pred_masks.view(B * N, -1)

            with autocast(enabled=False):
                pred_masks = pred_masks.float()
                tgt_masks = tgt_masks.float()
                pred_logits = pred_logits.float()
                mask_score = self.mask_score(pred_masks, tgt_masks)
                # matching_prob = pred_logits.view(B * N, -1)[:, tgt_ids]
                # cv 0527
                matching_prob = pred_logits.reshape(B * N, -1)[:, tgt_ids]

                C = (mask_score ** self.alpha) * (matching_prob ** self.beta)

            C = C.view(B, N, -1).cpu()
            # hungarian matching
            sizes = [len(v.unique())-1 for v in targets] # 不包括背景
            indices = [linear_sum_assignment(c[i], maximize=True)
                       for i, c in enumerate(C.split(sizes, -1))]
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
                j, dtype=torch.int64)) for i, j in indices]
            return indices, tgt_masks

class Cal_Loss(nn.Module):
    def __init__(self):
        super(Cal_Loss, self).__init__()

    def forward(self, outputs, targets):
        # matcher
        matcher = InstMatcher()
        indices, target_masks = matcher(outputs, targets)
        num_instances = torch.as_tensor([sum(len(v.unique())-1 for v in targets)], dtype=torch.float)

        # loss_labels
        src_logits = outputs['pred_logits']
        idx = get_src_permutation_idx(indices)
        targets_labels = [target.unique() for target in targets]
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets_labels, indices)])
        target_classes = torch.full(src_logits.shape[:2], 14, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        target_classes = target_classes.flatten(0, 1)

        pos_inds = torch.nonzero(target_classes != 14, as_tuple=True)[0]  # 14 is the class number without background
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1

        loss_labels = sigmoid_focal_loss(src_logits, labels, alpha=0.25, gamma=2.0, reduction="sum") / num_instances.cuda()

        # loss_objectness
        src_idx = get_src_permutation_idx(indices)
        tgt_idx = get_tgt_permutation_idx(indices)
        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]

        src_masks = src_masks[src_idx]
        src_masks = src_masks.flatten(1)
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])
        cum_sum = 0
        num_masks = [len(v.unique()) - 1 for v in targets]
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]
        target_masks = target_masks[mix_tgt_idx].flatten(1)

        ious = compute_mask_iou(src_masks, target_masks)

        tgt_iou_scores = ious
        src_iou_scores = src_iou_scores[src_idx]
        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)


        loss_objectness = F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')
        # loss_masks
        loss_dice = dice_loss(src_masks, target_masks) / num_instances.cuda()
        loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        # loss_masks = 1.0 * loss_dice + 2.0 * loss_mask
        loss_masks = 2.0 * loss_dice + 5.0 * loss_mask

        # loss_all = loss_labels + loss_objectness + loss_masks
        # lambda
        loss_all = 2.0 * loss_labels + 1.0 * loss_masks + 1.0 * loss_objectness
        return loss_all


class Cal_IOU(nn.Module):
    def __init__(self):
        super(Cal_IOU, self).__init__()

    def forward(self, outputs, targets):
        # matcher
        matcher = InstMatcher()
        indices, target_masks = matcher(outputs, targets)

        src_idx = get_src_permutation_idx(indices)
        tgt_idx = get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]

        src_masks = src_masks[src_idx]
        src_masks = src_masks.flatten(1)
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])
        cum_sum = 0
        num_masks = [len(v.unique()) - 1 for v in targets]
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]
        target_masks = target_masks[mix_tgt_idx].flatten(1)

        ious = compute_mask_iou(src_masks, target_masks)
        miou = sum(ious)/len(ious)

        return miou

if __name__ == "__main__":
    pred_logits = torch.rand(8,30,15)
    pred_masks = torch.rand(8,30,10000)
    pred_scores = torch.rand(8,30,1)
    targets = torch.randint(0, 15, (8, 10000)).long()

    outputs = {
                "pred_logits": pred_logits,
                "pred_masks": pred_masks,
                "pred_scores": pred_scores,
            }
    loss_all = Cal_Loss()(outputs, targets)
    print(loss_all)