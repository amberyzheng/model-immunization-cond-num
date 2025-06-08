from timm.models import create_model
from torch import nn

class ViT(nn.Module):
    def __init__(self, pretrained=True):
        super(ViT, self).__init__()
        self.model = create_model("vit_base_patch16_224", pretrained=pretrained)
        self.model.classifier = None
        self.head_drop = self.model.head_drop
        self.pool = self.model.pool
        self.fc_norm = self.model.fc_norm

        # Freeze all parameters by default
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Unfreeze specific parts by matching names
        for name, param in self.model.named_parameters():
            if "blocks.11" in name or "norm" in name or "head" in name:
                param.requires_grad = True
        

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)

        return x