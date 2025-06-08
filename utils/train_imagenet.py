import sys
sys.path.append('..')
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import condition_number


def get_feature_classifier(model):
    """
    Splits the given model into a feature extractor and a classifier.

    Args:
        model (torch.nn.Module): The model to split (e.g., ResNet or VisionTransformer).

    Returns:
        feature_extractor (torch.nn.Module): The feature extractor part of the model.
        classifier (torch.nn.Module): The classifier part of the model.
    """
    if type(model).__name__ == "ResNet":
        # For ResNet, remove the fc layer to get the feature extractor
        feature_extractor = nn.Sequential(*list(model.children())[:-1])  # All layers except fc
        classifier = model.fc  # The fully connected (fc) layer

    elif type(model).__name__ == "VisionTransformer":
        # For VisionTransformer, remove the head to get the feature extractor
        class VisionTransformerFeatureExtractor(nn.Module):
            def __init__(self, vit_model):
                super(VisionTransformerFeatureExtractor, self).__init__()
                self.patch_embed = vit_model.patch_embed
                self.pos_drop = vit_model.pos_drop
                self.blocks = vit_model.blocks
                self.norm = vit_model.norm  # Normalization before the head

            def forward(self, x):
                x = self.patch_embed(x)
                x = self.pos_drop(x)
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)  # Output features
                return x

        feature_extractor = VisionTransformerFeatureExtractor(model)
        classifier = model.head  # The head is the classification layer

    else:
        raise ValueError(f"Unsupported model type: {type(model).__name__}")

    return feature_extractor, classifier


class ImmuLoss(nn.Module):
    def __init__(self, loss_fn, lambda_r1=1.0, lambda_r2=1.0, lambda_kappa=1.0, lambda_reg=1.0, loss_type="upper_bound_sgd"):
        """
        Initialize the ImmuLoss class.

        Args:
            config (dict): Configuration dictionary for the model.
            lambda_r1, lambda_r2, lambda_kappa, lambda_reg (float): Regularization coefficients.
            loss_type (str): Type of loss to use ("kappa", "upper_bound_sgd", etc.).
        """
        super(ImmuLoss, self).__init__()
        self.lambda_r1 = lambda_r1
        self.lambda_r2 = lambda_r2
        self.lambda_kappa = lambda_kappa
        self.lambda_reg = lambda_reg
        self.loss_type = loss_type
        self.loss_fn = loss_fn
        self.imma_epoch = 0
        self.loss_values = []

    def classification_loss(self, logits, y):
        """
        Compute classification loss.

        Args:
            logits (Tensor): Model output logits.
            y (Tensor): Ground truth labels.

        Returns:
            Tensor: Classification loss.
        """
        return self.loss_fn(logits, y)

    def kappa_loss(self, X1_phi, X2_phi):
        """
        Compute kappa loss.

        Args:
            X1_phi (Tensor): Extracted features for dataset 1.
            X2_phi (Tensor): Extracted features for dataset 2.

        Returns:
            Tensor: Kappa loss.
        """
        A1_phi = X1_phi.T @ X1_phi
        kappa1 = condition_number(A1_phi)

        A2_phi = X2_phi.T @ X2_phi
        kappa2 = condition_number(A2_phi)

        k1 = kappa1
        k2 = 1 / kappa2

        self.log("kappa1", k1)
        self.log("kappa2", k2)

        return self.lambda_r2 * k2 + self.lambda_r1 * k1

    def upper_bound_loss(self, X1_phi, X2_phi):
        """
        Compute upper bound loss.

        Args:
            X1_phi (Tensor): Extracted features for dataset 1.
            X2_phi (Tensor): Extracted features for dataset 2.

        Returns:
            Tensor: Upper bound loss.
        """
        K = X1_phi.size(-1)

        S1 = X1_phi.T @ X1_phi
        S2 = X2_phi.T @ X2_phi

        _, s1_max, cond_s1 = condition_number(S1, return_eigvals=True)
        s2_min, _, cond_s2 = condition_number(S2, return_eigvals=True)

        n = 2
        r1 = s1_max**2 / 2 - torch.norm(S1, p="fro")**2 / (2 * K)
        r2 = 1 / (torch.norm(S2, p="fro")**n / (2 * K) - s2_min**n / 2)

        self.log("r1", r1)
        self.log("r2", r2)
        self.log("cond_S1", cond_s1)
        self.log("cond_S2", cond_s2)

        return self.lambda_r1 * r1 + self.lambda_r2 * r2

    def forward(self, X1_phi, y1, X2_phi, y2, classifier):
        """
        Compute the total loss.

        Args:
            X1_phi (Tensor): Extracted features for dataset 1.
            y1 (Tensor): Ground truth for dataset 1.
            X2_phi (Tensor): Extracted features for dataset 2.
            y2 (Tensor): Ground truth for dataset 2.
            classifier (Tensor): Classifier weights or logits.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Total loss, reconstruction error, and regularization loss.
        """
        # Compute reconstruction error
        reconstruction_error = self.classification_loss(classifier(X1_phi), y1)

        # Compute regularization loss
        if self.loss_type == "kappa":
            regularization_loss = self.kappa_loss(X1_phi, X2_phi)
        elif self.loss_type == "upper_bound_sgd":
            regularization_loss = self.upper_bound_loss(X1_phi, X2_phi)
        elif self.loss_type == "max_only":
            self.lambda_r1 = 0
            regularization_loss = self.upper_bound_loss(X1_phi, X2_phi)
        elif self.loss_type == "imma":
            recon2 = self.classification_loss(X2_phi @ classifier, y2)
            self.imma_epoch += 1
            if self.imma_epoch // 2 == 0:
                return reconstruction_error + recon2, reconstruction_error + recon2, 0
            else:
                return reconstruction_error - recon2, reconstruction_error - recon2, 0
        elif self.loss_type == "classification":
            regularization_loss = 0
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Use 'kappa'.")

        # Total loss
        total_loss_value = reconstruction_error + self.lambda_kappa * regularization_loss
        self.loss_values.append(total_loss_value.item())

        return total_loss_value, reconstruction_error, regularization_loss

    def log(self, name, value):
        """
        Log the value (example implementation for extensibility).

        Args:
            name (str): Name of the value.
            value (float or Tensor): Value to log.
        """
        print(f"{name}: {value.item() if isinstance(value, torch.Tensor) else value}")