from random import randint
import torch

scaler = torch.cuda.amp.GradScaler()


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

    queue_length = len(queue)
    losses = 0

    j = randint(0, queue_length - 1)

    features = []
    labels = []

    # model.eval()
    # with torch.no_grad():
    for i in range(queue_length):
        support_data = queue[i]["batch"]["support"]
        support_task = queue[i]["task"]
        with torch.cuda.amp.autocast():
            _, support_features = model.forward(
                support_task, support_data, classify=False
            )
        features.append(scaler.scale(support_features))
        labels.append(support_data["label"])

    features = torch.cat(features)
    labels = torch.cat(labels).to(device)
    prototypes = compute_prototypes(features, labels)

    query_data = queue[j]["batch"]["query"]
    query_task = queue[j]["task"]

    query_labels = query_data["label"].to(device)

    with torch.cuda.amp.autocast():
        query_outputs, query_features = model.forward(query_task, query_data)
        loss = criterion(query_features, query_outputs, query_labels, prototypes)

    # loss.backward()
    scaler.scale(loss).backward()
    losses += loss.detach().item()

    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    # optim.step()
    scaler.step(optim)
    scaler.update()

    return losses / queue_length

