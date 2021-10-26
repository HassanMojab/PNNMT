import torch
import torch.nn as nn
import torch.nn.functional as F

compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)


def euclidean_dist(x, y):
    """
  Compute euclidean distance between two tensors
  """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


## prototype loss (PL): "Robust Classification with Convolutional Prototype Learning"
class PrototypeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, labels, prototypes):
        n = features.shape[0]
        seen_labels = torch.unique(labels)

        prototype_dic = {
            l.item(): prototypes[idx].reshape(1, -1)
            for idx, l in enumerate(seen_labels)
        }
        loss = 0.0
        for idx, feature in enumerate(features):
            dists = euclidean_dist(
                feature.reshape(1, -1), prototype_dic[labels[idx].item()]
            )  # [q_num, cls_num]
            loss += dists

        loss /= n
        return loss


class DCELoss(nn.Module):
    def __init__(self, device, gamma=0.05):
        super().__init__()
        self.gamma = gamma
        self.device = device

    def forward(self, features, prototypes, args):
        n_classes = args.ways
        n_query = args.query_num + args.target_shot

        dists = euclidean_dist(features, prototypes)
        # dists = (-self.gamma * dists).exp()

        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
        target_inds = torch.arange(0, n_classes, device=self.device, dtype=torch.long)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        return loss_val


class CPELoss(nn.Module):
    def __init__(self, device, args):
        super().__init__()
        self.args = args

        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3

        self.dce = DCELoss(device, gamma=args.temp_scale)
        self.proto = PrototypeLoss()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, features, outputs, labels, prototypes):
        dce_loss = self.dce(features, prototypes, self.args)
        cls_loss = self.ce(outputs, labels.long())
        prototype_loss = self.proto(features, labels, prototypes)
        return (
            self.lambda_1 * dce_loss
            + self.lambda_2 * cls_loss
            + self.lambda_3 * prototype_loss
        )
