from typing import Tuple

import torch
from torch import nn, Tensor


class NormLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(NormLinear, self).__init__(in_features, out_features, bias=False)

    def forward(self, inp: Tensor) -> Tensor:
        return nn.functional.linear(nn.functional.normalize(inp),
                                    nn.functional.normalize(self.weight))


class CircleLossLikeCE(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLossLikeCE, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inp: Tensor, label: Tensor) -> Tensor:
        a = torch.clamp_min(inp + self.m, min=0).detach()
        src = torch.clamp_min(
            - inp.gather(dim=1, index=label.unsqueeze(1)) + 1 + self.m,
            min=0,
        ).detach()
        a.scatter_(dim=1, index=label.unsqueeze(1), src=src)

        sigma = torch.ones_like(inp, device=inp.device, dtype=inp.dtype) * self.m
        src = torch.ones_like(label.unsqueeze(1), dtype=inp.dtype, device=inp.device) - self.m
        sigma.scatter_(dim=1, index=label.unsqueeze(1), src=src)

        return self.loss(a * (inp - sigma) * self.gamma, label)


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLossBackward(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLossBackward, self).__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = torch.log(1 + torch.clamp_max(torch.exp(logit_n).sum() * torch.exp(logit_p).sum(), max=1e38))
        z = - torch.exp(- loss) + 1

        """
        Eq. 10:
        sp.backward(gradient=z * (- ap) * torch.softmax(- logit_p, dim=0) * self.gamma, retain_graph=True)
        I modified it to 
        sp.backward(gradient=z * (- ap) * torch.softmax(logit_p, dim=0) * self.gamma, retain_graph=True)
        """

        sp.backward(gradient=z * (- ap) * torch.softmax(logit_p, dim=0) * self.gamma, retain_graph=True)
        sn.backward(gradient=z * an * torch.softmax(logit_n, dim=0) * self.gamma)

        return loss.detach()


if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
    lbl = torch.randint(high=10, size=(256,))

    inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

    circle_loss_backward = CircleLossBackward(m=0.25, gamma=80)
    circle_loss = circle_loss_backward(inp_sp, inp_sn)

    print(circle_loss)
