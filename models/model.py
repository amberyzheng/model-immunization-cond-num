import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import gc
from models.feature_extractor import LinearFeatureExtractor, PretrainedFeatureExtractor
from utils.loss import condition_number, LabelSmoothingCrossEntropy
from utils.option import get_config_value

class Model(pl.LightningModule):
    def __init__(self, config):
        """
        Initialize the model with parameters from the configuration.
        Args:
            config (dict): Model-specific configuration loaded from `model.yaml`.
        """
        super().__init__()
        self.save_hyperparameters(config)
        
        self.config = config
        self.feat_size = config["feat_size"]
        self.num_classes = 1000 if "pt" in config["feature_extractor_type"] else 1
        self.lambda_r1 = get_config_value(config, "lambda_r1")
        self.lambda_r2 = get_config_value(config, "lambda_r2")
        self.lambda_reg = float(config.get("lambda_reg", 1.0))
        self.lr = float(config["lr"])
        self.ckpt_path = config.get("ckpt_path", None)
        self.use_precond = True
        self.num_epochs = config.get("max_epochs", 100)
        self.loss_values = []

        # Select feature extractor
        if config["feature_extractor_type"] == "linear":
            self.feature_extractor = LinearFeatureExtractor(self.feat_size, self.feat_size, self.ckpt_path, self.use_precond)
        elif "pt" in config["feature_extractor_type"]:
            if config["feature_extractor_type"] == "pt_imagenet":
                self.feature_extractor = PretrainedFeatureExtractor(dataset="imagenet")
            elif config["feature_extractor_type"] == "pt_imagenet_vit":
                self.feature_extractor = PretrainedFeatureExtractor(dataset="imagenet_vit")
        else:
            raise ValueError("Undefined feature_extractor_type.")

        if 'pt' in config['feature_extractor_type']:
            self.classifier = self.feature_extractor.classifier.to(dtype=torch.double)
        else:
            self.classifier = nn.Parameter(torch.randn(self.feat_size, self.num_classes, dtype=torch.double).squeeze(-1))

    def forward(self, x):
        return self.feature_extractor(x)

    def regression_loss(self, X, y):
        X_phi = self.feature_extractor(X)
        reconstruction_error = torch.norm(y.float() - X_phi @ self.classifier, 2) ** 2 / y.size(0)
        return reconstruction_error
    
    def classification_loss(self, X, y):
        if "pt" in self.config["feature_extractor_type"]:
            logits = self.classifier(self.feature_extractor(X))
            classification_loss = LabelSmoothingCrossEntropy()(logits, y.long())
        else:
            logits = self.feature_extractor(X) @ self.classifier 
            logits = logits.squeeze(-1) 
            y = y.float()
            classification_loss = F.binary_cross_entropy_with_logits(logits, y)
        return classification_loss

    def reg_loss(self, X1, X2):
        K = X1.size(-1)
        if self.config["feature_extractor_type"] == "linear":
            X1_phi = self.feature_extractor(X1, lambda_reg=1, use_precond=True)
        else:
            X1_phi = self.feature_extractor(X1)
        S1 = X1_phi.T @ X1_phi

        if self.config["feature_extractor_type"] == "linear":
            X2_phi = self.feature_extractor(X2, lambda_reg=self.lambda_reg, use_precond=True)
        else:
            X2_phi = self.feature_extractor(X2)
        S2 = X2_phi.T @ X2_phi

        _, s1_max, cond_s1 = condition_number(S1, return_eigvals=True)
        s2_min, _, cond_s2 = condition_number(S2, return_eigvals=True)

        n = 2
        r1 = s1_max**2 / 2 - torch.norm(S1, p="fro")**2 / (2 * K)
        r2 = 1 / (torch.norm(S2, p="fro")**n / (2 * K) - s2_min**n / 2)

        self.log("r1", r1.detach(), prog_bar=False, on_epoch=True)
        self.log("r2", r2.detach(), prog_bar=False, on_epoch=True)
        self.log("cond_S1", cond_s1.detach(), prog_bar=False, on_epoch=True)
        self.log("cond_S2", cond_s2.detach(), prog_bar=False, on_epoch=True)

        return self.lambda_r1 * r1 + self.lambda_r2 * r2

    def total_loss(self, X1, y1, X2, y2, current_epoch):
        if self.config['task'] == 'regression':
            reconstruction_error = self.regression_loss(X1, y1)
        else:
            reconstruction_error = self.classification_loss(X1, y1)

        regularization_loss = self.reg_loss(X1, X2)
        total_loss_value = reconstruction_error + regularization_loss
        self.loss_values.append(total_loss_value.item())

        return total_loss_value, reconstruction_error, regularization_loss

    def training_step(self, batch, batch_idx):
        (X1, y1), (X2, y2) = batch
        X1 = X1.to(dtype=torch.double)
        X2 = X2.to(dtype=torch.double)
        y1 = y1.to(dtype=torch.double)
        y2 = y2.to(dtype=torch.double)
        
        total_loss, reconstruction_error, regularization_loss = self.total_loss(
            X1, y1, X2, y2, current_epoch=self.current_epoch
        )

        self.log("train_loss", total_loss.detach(), prog_bar=True, on_epoch=True)
        self.log("reconstruction_loss", reconstruction_error.detach(), prog_bar=False, on_epoch=True)
        self.log("regularization_loss", regularization_loss.detach(), prog_bar=False, on_epoch=True)

        torch.cuda.empty_cache()
        gc.collect()
        
        return total_loss

    def get_loss_values(self):
        return self.loss_values

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()

    def configure_optimizers(self):
        if 'pt' in self.config['feature_extractor_type']:
            feature_extractor_params = self.feature_extractor.feature_extractor.parameters()
            classifier_params = self.classifier.parameters()
        else:
            feature_extractor_params = self.feature_extractor.parameters()
            classifier_params = [self.classifier]

        if 'pt' in self.config['feature_extractor_type']:
            opt = torch.optim.SGD([
                {'params': feature_extractor_params, 'lr': self.lr, 'weight_decay': 0.0},
                {'params': classifier_params, 'lr': self.lr, 'weight_decay': 2e-05}
            ], lr=0.05, momentum=0.9, nesterov=True)
            
            for group in opt.param_groups:
                group['initial_lr'] = 0.05
                
            if hasattr(self.feature_extractor, 'frozen_extractor'):
                for param in self.feature_extractor.frozen_extractor.parameters():
                    param.requires_grad = False

            scheduler_kwargs = {
                'sched': 'cosine',
                'num_epochs': 1,
                'decay_epochs': 90,
                'decay_milestones': [90, 180, 270],
                'warmup_epochs': 5,
                'cooldown_epochs': 0,
                'patience_epochs': 10,
                'decay_rate': 0.1,
                'min_lr': 0,
                'warmup_lr': 1e-5,
                'warmup_prefix': False,
                'noise': None,
                'noise_pct': 0.67,
                'noise_std': 1.0,
                'noise_seed': 42,
                'cycle_mul': 1.0,
                'cycle_decay': 0.5,
                'cycle_limit': 1,
                'k_decay': 1.0,
                'plateau_mode': 'max',
                'step_on_epochs': True
            }

            from timm.scheduler import CosineLRScheduler
            self.scheduler = CosineLRScheduler(
                opt,
                t_initial=scheduler_kwargs['decay_epochs'],
                lr_min=scheduler_kwargs['min_lr'],
                warmup_t=scheduler_kwargs['warmup_epochs'],
                warmup_lr_init=scheduler_kwargs['warmup_lr'],
                warmup_prefix=scheduler_kwargs['warmup_prefix'],
                noise_range_t=scheduler_kwargs['noise'],
                noise_pct=scheduler_kwargs['noise_pct'],
                noise_std=scheduler_kwargs['noise_std'],
                noise_seed=scheduler_kwargs['noise_seed'],
                k_decay=scheduler_kwargs['k_decay'],
            )
        else:
            opt = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=lambda epoch: 0.1 if epoch >= self.num_epochs // 2 else 1.0
            )

        return {"optimizer": opt}
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        self.scheduler.step(self.current_epoch + self.global_step / self.trainer.estimated_stepping_batches) 