import oneflow.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        inputs = self.flatten(inputs)
        logits = self.linear_relu_stack(inputs)

        if labels is not None and self.training:
            losses = self.loss_func(logits, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": logits}


from libai.models.bert_model import BertModel
import oneflow.nn as nn


class BERT_Classifier(nn.Module):
    def __init__(self, label_num):
        super().__init__()
        self.encoder = BertModel.from_pretrained('./bert-base-chinese')
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.fc = nn.Linear(768, label_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)[0]
        x = encoder_output[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        if tokentype_ids == None:
            return None, x
        else:
            return self.criterion(x, tokentype_ids), x

        if labels is not None and self.training:
            losses = self.loss_func(logits, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": logits}