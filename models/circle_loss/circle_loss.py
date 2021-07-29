from typing import Tuple

import torch
from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)

    # print('similarity_matrix = ', similarity_matrix)

    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    # print(label_matrix)
    positive_matrix = label_matrix.triu(diagonal=1)
    # print('positive_matrix = ', positive_matrix)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)
    # print('negative_matrix = ', negative_matrix)

    similarity_matrix = similarity_matrix.view(-1)
    # print(similarity_matrix)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    # print('similarity_matrix[positive_matrix] = ', similarity_matrix[positive_matrix])
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(256, 1, requires_grad=True))
    feat = torch.FloatTensor([0.9, 0.2, 0.3, 0.5, 0.1, 0.8]).view(6, 1)
    lbl = torch.LongTensor([1, 0, 1, 1, 0, 1])
    print(feat.size(), lbl.size())

    inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

    criterion = CircleLoss(m=0.25, gamma=256)
    circle_loss = criterion(inp_sp, inp_sn)

    print(circle_loss)
