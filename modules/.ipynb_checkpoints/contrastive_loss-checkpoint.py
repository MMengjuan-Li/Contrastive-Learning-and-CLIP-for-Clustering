import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# class SoftContrastiveLoss(nn.Module):
#     """
#     Implementation of Soft Contrastive Loss (SCL) with the correct handling of denominator.
#     """
#     def __init__(self, tau=0.5):
#         super(SoftContrastiveLoss, self).__init__()
#         self.tau = tau

#     def forward(self, z, pseudo_labels, confidences):
#         """
#         Args:
#             z (torch.Tensor): Feature embeddings of shape (N, D).
#             pseudo_labels (torch.Tensor): Pseudo labels of shape (N,).
#             confidences (torch.Tensor): Confidence scores of the pseudo labels, shape (N,).

#         Returns:
#             loss (torch.Tensor): Soft contrastive loss.
#         """
#         eps = 1e-8  # Small value for numerical stability

#         # Compute pairwise similarity
#         sim = torch.mm(z, z.t())  # Shape: (N, N)
#         sim = sim / self.tau  # Scale similarity with temperature

#         # Mask for removing self-similarity
#         N = z.size(0)
#         mask_self = torch.eye(N, dtype=torch.bool, device=z.device)  # Diagonal mask for self-similarity
#         sim = sim.masked_fill(mask_self, float('-inf'))  # 将相似度矩阵中所有对角线位置的值（即样本与自身的相似度）设置为负无穷大

#         # Positive pair mask (same pseudo labels)
#         pos_mask = pseudo_labels.unsqueeze(1) == pseudo_labels.unsqueeze(0)  # Shape: (N, N)

#         # Negative pair mask (different pseudo labels)
#         neg_mask = ~pos_mask  # Negate the positive mask

#         # Compute positive logits (confidence-weighted)
#         pos_logits = sim * pos_mask.float() * (confidences.unsqueeze(1) + eps)  # Shape: (N, N)

#         # Compute numerator: exp of positive logits (only positive pairs contribute)
#         num = torch.exp(pos_logits).sum(dim=1, keepdim=True) + eps  # Shape: (N, 1)

#         # Compute denominator: exp of negative logits (only negative pairs contribute)
#         denom = torch.exp(sim * neg_mask.float()).sum(dim=1, keepdim=True) + eps  # Shape: (N, 1)

#         # Soft contrastive loss
#         loss = -torch.log(num / denom).mean()

#         return loss


class SoftContrastiveLoss(nn.Module):
    """
    Implementation of Soft Contrastive Loss (SCL) with numerical stability improvements.
    """
    def __init__(self, tau=0.5):
        super(SoftContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, z, pseudo_labels, confidences):
        """
        Args:
            z (torch.Tensor): Feature embeddings of shape (N, D).
            pseudo_labels (torch.Tensor): Pseudo labels of shape (N,).
            confidences (torch.Tensor): Confidence scores of the pseudo labels, shape (N,).

        Returns:
            loss (torch.Tensor): Soft contrastive loss.
        """
        eps = 1e-8  # Small value for numerical stability

        # Compute pairwise similarity
        sim = torch.mm(z, z.t())  # Shape: (N, N)


        # Scale similarity with temperature
        sim = sim / self.tau

        # Mask for removing self-similarity
        N = z.size(0)
        mask = torch.eye(N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float('-inf'))

        # Clamp similarity values to avoid overflow in exp()
        # sim = torch.clamp(sim, min=-20, max=20)

        # Compute positive logits
        pos_mask = pseudo_labels.unsqueeze(1) == pseudo_labels.unsqueeze(0)  # Positive pair mask
        pos_logits = sim * pos_mask.float() * (confidences.unsqueeze(1) + eps)  # Confidence-weighted similarity

        # Compute denominator (all pairs except self-pairs)
        denom = torch.exp(sim).sum(dim=1, keepdim=True) + eps  # Shape: (N, 1)

        # Numerator (sum of positive logits)
        num = torch.exp(pos_logits).sum(dim=1, keepdim=True) + eps  # Shape: (N, 1)

        # Soft contrastive loss
        loss = -torch.log(num / denom).mean()

        return loss





class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
