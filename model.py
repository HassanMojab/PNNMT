import torch
import torch.nn as nn
from transformers import AutoModel, logging


logging.set_verbosity_error()


class BertMetaLearning(nn.Module):
    def __init__(self, args):
        super(BertMetaLearning, self).__init__()
        self.args = args
        self.device = None

        self.model = AutoModel.from_pretrained(
            args.model_name, local_files_only=args.local_model
        )

        # Sequence Classification
        self.sc_dropout = nn.Dropout(args.dropout)
        self.sc_classifier = nn.Linear(args.hidden_dims, args.sc_labels)

    def forward(self, task, data, classify=True):
        data["input_ids"] = data["input_ids"].to(self.device)
        data["attention_mask"] = data["attention_mask"].to(self.device)
        data["token_type_ids"] = data["token_type_ids"].to(self.device)
        data["label"] = data["label"].to(self.device)

        outputs = self.model(
            data["input_ids"],
            attention_mask=data["attention_mask"],
            token_type_ids=data["token_type_ids"].to(torch.long),
        )

        features = outputs[1]  # [n, 768]

        logits = None

        if classify:
            pooled_output = self.sc_dropout(features)
            logits = self.sc_classifier(pooled_output)

        # loss = F.cross_entropy(logits, data["label"], reduction="none")
        # outputs = (loss, logits) + outputs[2:]

        return logits, features

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = args[0]  # store device
        self.model = self.model.to(*args, **kwargs)
        self.sc_dropout = self.sc_dropout.to(*args, **kwargs)
        self.sc_classifier = self.sc_classifier.to(*args, **kwargs)
        return self
