import torch
import torch.nn.functional as F


def evaluateNLI(model, data, device, return_matrix=False):
    with torch.no_grad():
        total_loss = 0.0
        correct = 0.0
        total = 0.0
        matrix = [[0 for _ in range(3)] for _ in range(3)]
        for batch in data:
            logits, _ = model.forward(batch)
            batch["label"] = batch["label"].to(device)
            loss = F.cross_entropy(logits, batch["label"].long(), reduction="none")
            loss = loss.mean()

            prediction = torch.argmax(logits, dim=1)
            correct += torch.sum(prediction == batch["label"]).item()
            for k in range(batch["label"].shape[0]):
                matrix[batch["label"][k]][prediction[k]] += 1
            total += batch["label"].shape[0]
            total_loss += loss.detach().item()

        total_loss /= len(data)
        total_acc = correct / total

        if return_matrix:
            return total_loss, total_acc, matrix
        return total_loss, total_acc
