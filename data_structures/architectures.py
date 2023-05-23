from torch import nn
from transformers import BertForTokenClassification
from transformers import RobertaForTokenClassification
from transformers import XLNetForTokenClassification


class XLNetModel(nn.Module):

    def __init__(self, pretrained_model, unique_labels):

        super(XLNetModel, self).__init__()

        self.xlnet = XLNetForTokenClassification.from_pretrained(pretrained_model, num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.xlnet(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output


class RobertaModel(nn.Module):
    def __init__(self, pretrained_model, unique_labels):

        super(RobertaModel, self).__init__()

        self.roberta = RobertaForTokenClassification.from_pretrained(
            pretrained_model, num_labels=len(unique_labels)
        )

    def forward(self, input_id, mask, label):

        output = self.roberta(
            input_ids=input_id, attention_mask=mask, labels=label, return_dict=False
        )

        return output


class BertModel(nn.Module):
    def __init__(self, pretrained_model, unique_labels):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(
            pretrained_model, num_labels=len(unique_labels)
        )

    def forward(self, input_id, mask, label):

        output = self.bert(
            input_ids=input_id, attention_mask=mask, labels=label, return_dict=False
        )
        return output