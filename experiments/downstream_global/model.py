"""BigBird encoder + mean pooling + linear classifier."""

import torch.nn as nn
from transformers import BigBirdModel, BigBirdPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BigBirdMeanPoolClassifier(BigBirdPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BigBirdModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                output_hidden_states=None, **kwargs):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask,
            output_hidden_states=output_hidden_states, **kwargs,
        )
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
