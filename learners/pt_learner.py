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


def pt_learner(model, queue, criterion, optim, args, device):
    model.train()
    optim.zero_grad()

    queue_len = len(queue)

    data = {
        "input_ids": torch.cat(
            [item["support"]["input_ids"] for item in queue]
            + [item["query"]["input_ids"] for item in queue]
        ),
        "attention_mask": torch.cat(
            [item["support"]["attention_mask"] for item in queue]
            + [item["query"]["attention_mask"] for item in queue]
        ),
    }
    labels = torch.cat(
        [item["support"]["label"] for item in queue]
        + [item["query"]["label"] for item in queue]
    ).to(device)

    outputs, features = model.forward(data)

    prototypes = compute_prototypes(
        features[: len(queue_len * args.shot)], labels[: len(queue_len * args.shot)]
    )

    loss = criterion(
        features[len(queue_len * args.shot) :],
        outputs[len(queue_len * args.shot) :],
        labels[len(queue_len * args.shot) :],
        prototypes,
    )

    # support_features = []
    # support_labels = []

    # query_outputs = []
    # query_features = []
    # query_labels = []

    # for item in queue:
    #     support_data = item["support"]
    #     query_data = item["query"]

    #     _, features = model.forward(support_data, classify=False)
    #     support_features.append(features)

    #     outputs, features = model.forward(query_data)
    #     query_outputs.append(outputs)
    #     query_features.append(features)

    #     support_labels.append(support_data["label"])
    #     query_labels.append(query_data["label"])

    # support_features = torch.cat(support_features)
    # support_labels = torch.cat(support_labels).to(device)
    # prototypes = compute_prototypes(support_features, support_labels)

    # query_outputs = torch.cat(query_outputs)
    # query_features = torch.cat(query_features)
    # query_labels = torch.cat(query_labels).to(device)

    # loss = criterion(query_features, query_outputs, query_labels, prototypes)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optim.step()

    return loss.detach().item()
