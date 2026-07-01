import copy

import torch
import torchvision
from nltk.tokenize import wordpunct_tokenize
from torch import nn
from transformers import AutoModel, AutoTokenizer

from merlin.models import i3res


class ImageEncoder(nn.Module):
    def __init__(
        self,
        ImageEmbedding: bool = False,
        HCCClassification: bool = False,
        PhenotypeCls: bool = False,
        rotate_flip=False,
    ):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.HCCClassification = HCCClassification
        self.PhenotypeCls = PhenotypeCls
        self.rotate_flip = rotate_flip
        resnet = torchvision.models.resnet152(pretrained=True)
        self.i3_resnet = i3res.I3ResNet(
            copy.deepcopy(resnet),
            class_nb=1692,
            conv_class=True,
            ImageEmbedding=self.ImageEmbedding,
            HCCClassification=self.HCCClassification,
            PhenotypeCls=self.PhenotypeCls,
            rotate_flip=self.rotate_flip,
        )

    def forward(self, image):
        if self.ImageEmbedding:
            contrastive_features = self.i3_resnet(image)
            return contrastive_features
        elif self.HCCClassification:
            return self.i3_resnet(image)
        elif self.PhenotypeCls:
            return self.i3_resnet(image)
        else:
            contrastive_features, ehr_features, hcc_features = self.i3_resnet(image)
            return contrastive_features, ehr_features, hcc_features


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, 512)

    def forward(self, text_labels):
        text_labels = [sanitize_report(text) for text in text_labels]
        # print(f"Text labels: {text_labels}")
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.linear_layer(text_embeddings)
        return text_embeddings


class MerlinArchitecture(nn.Module):
    def __init__(
        self,
        init_logit_scale: float = 1.0,
        ImageEmbedding: bool = False,
        HCCClassification: bool = False,
        PhenotypeCls: bool = False,
        rotate_flip: bool = False,
    ):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.HCCClassification = HCCClassification
        self.PhenotypeCls = PhenotypeCls
        self.rotate_flip = rotate_flip
        self.encode_image = ImageEncoder(
            ImageEmbedding=self.ImageEmbedding,
            HCCClassification=self.HCCClassification,
            PhenotypeCls=self.PhenotypeCls,
            rotate_flip=self.rotate_flip,
        )
        self.encode_text = TextEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, image, text=None):
        if self.ImageEmbedding and text is None:
            image_features = self.encode_image(image)
            return image_features
        elif self.HCCClassification and text is None:
            hcc_probs = self.encode_image(image)
            return hcc_probs
        elif self.PhenotypeCls and text is None:
            phenotype_probs = self.encode_image(image)
            return phenotype_probs
        elif self.ImageEmbedding and text is not None:
            raise ValueError("Text input not required for image embedding")
        elif self.HCCClassification and text is not None:
            raise ValueError("Text input not required for HCC classification")
        elif self.PhenotypeCls and text is not None:
            raise ValueError("Text input not required for phenotype classification")
        elif text is None:
            raise ValueError("Text input required for Image and Text embedding")

        image_features, ehr_features, hcc_features = self.encode_image(image)
        text_features = self.encode_text(text)

        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return (
            image_features,
            ehr_features,
            hcc_features,
            text_features,
        )


def sanitize_report(report):
    report = report.lower()
    return " ".join(wordpunct_tokenize(report))
