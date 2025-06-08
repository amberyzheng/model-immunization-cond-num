import os
import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.option import load_yaml, save_config, create_experiment_dir, set_seed
from utils.log import save_loss_plot, log_and_save_condition_numbers, log_and_save_avg_condition_numbers
from utils.evaluate_imgnet import evaluate_model
from models.model import Model
from dataset import DataModule

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model with configurations.")
    parser.add_argument('--config', type=str, required=True, help="Path to base config file.")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--digit1', type=int, default=None)
    parser.add_argument('--digit2', type=int, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load configurations
    config = load_yaml(args.config)
    experiment_dir = create_experiment_dir(config, args.seed)
    save_config(config, os.path.join(experiment_dir, "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract parameters
    training_params = config['training']
    model_params = config['model']
    data_params = config['data']

    # Initialize wandb
    wandb.init(
        project=training_params['logger']['project_name'],
        config=config,
        name=f"{config['model']['feature_extractor_type']}_{args.seed}"
    )

    if args.ckpt_path is not None:
        model_params['ckpt_path'] = args.ckpt_path

    if data_params['dataset_name'] == 'mnist':
        assert args.digit1 is not None and args.digit2 is not None, "Please specify two digits for the experiment."
        experiment_dir = os.path.join(experiment_dir, f"digit_{args.digit1}_{args.digit2}")
        os.makedirs(experiment_dir, exist_ok=True)
        target_classes = [args.digit1, args.digit2]
    else:
        target_classes = None

    # Initialize data module
    data_module = DataModule(
        batch_size=data_params.get("batch_size", None),
        dataset_name=data_params.get("dataset_name", None),
        d1_path=data_params.get("d1_path", None),
        d2_path=data_params.get("d2_path", None),
        target_classes=target_classes,
        max_samples_per_class=data_params.get("max_samples_per_class", None),
        d2_name=data_params.get("d2_name", None)
    )

    # Initialize model
    model = Model(model_params).to(device, dtype=torch.double)

    # Initialize logger and callbacks
    logger = WandbLogger(project=training_params['logger']['project_name'])
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, "checkpoints"),
        filename="last",
        save_last=True,
        save_top_k=0
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=training_params['max_epochs'],
        devices=training_params['gpus'],
        precision=training_params['precision'],
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1
    )

    # Train the model
    trainer.fit(model, data_module)

    # Get loss values and save loss plot
    loss_values = model.get_loss_values() 
    model = model.to(device, dtype=torch.double)

    # Evaluate model
    if 'imagenet' in data_params['dataset_name']:
        k = 100
        n = 20
        with torch.no_grad():
            log_and_save_avg_condition_numbers(data_module, model.feature_extractor, model.feature_extractor, k=k, n=n, output_dir=experiment_dir, device=device, train=True)
            test_acc = evaluate_model(model, data_params['d1_path'])
            with open(os.path.join(experiment_dir, "test_acc.txt"), "a") as f:
                f.write(f"Test Accuracy: {test_acc:.2f}%\n")
    else:
        (X1, y1), (X2, y2) = data_module.get_full_data(train=True)
        X1 = X1.to(dtype=torch.double, device=device)
        X2 = X2.to(dtype=torch.double, device=device)
        
        with torch.no_grad():
            X1_immu = model.feature_extractor(X1)
            X2_immu = model.feature_extractor(X2)
            log_and_save_condition_numbers(X1, X2, X1_immu, X2_immu, experiment_dir)