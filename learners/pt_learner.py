import torch


def compute_prototypes(
    support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
    """
  Compute class prototypes from support features and labels
  Args:
    support_features: for each instance in the support set, its feature vector
    support_labels: for each instance in the support set, its label
  Returns:
    for each label of the support set, the average feature vector of instances with this label
  """
    seen_labels = torch.unique(support_labels)

    # TODO: does it need to sort by labels??

    # Prototype i is the mean of all instances of features corresponding to labels == i
    return torch.cat(
        [
            support_features[(support_labels == l).nonzero(as_tuple=True)[0]]
            .mean(0)
            .reshape(1, -1)
            for l in seen_labels
        ]
    )


class PtLearner:
    def __init__(self, criterion, device):
        self.criterion = criterion
        self.device = device

        self.prototypes = None

    def train(self, model, queue, trg_queue, optim, iteration, args):
        model.train()
        optim.zero_grad()

        queue_len = len(queue)
        support_len = queue_len * args.shot * args.ways
        print("support_len:", support_len)

        data_list = [item["support"] for item in queue] + [
            item["query"] for item in queue + trg_queue
        ]
        print("data_list:", data_list)
        print("data_list len:", len(data_list))

        data = {
            "input_ids": torch.cat([item["input_ids"] for item in data_list]),
            "attention_mask": torch.cat([item["attention_mask"] for item in data_list]),
        }
        labels = torch.cat([item["label"] for item in data_list]).to(self.device)

        outputs, features = model.forward(data)
        new_prototypes = compute_prototypes(
            features[:support_len], labels[:support_len]
        )

        beta = args.beta * iteration / args.meta_iteration

        if iteration > 1 and beta > 0.0:
            self.prototypes = beta * self.prototypes + (1 - beta) * new_prototypes
        else:
            self.prototypes = new_prototypes

        print("query features:", features[support_len:])
        print("query features shape:", features[support_len:].shape)

        loss = self.criterion(
            features[support_len:],
            outputs[support_len:],
            labels[support_len:],
            self.prototypes,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        if beta > 0.0:
            self.prototypes = self.prototypes.detach()
        else:
            self.prototypes = None

        return loss.detach().item()
