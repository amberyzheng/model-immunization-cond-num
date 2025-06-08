import os
import matplotlib.pyplot as plt
import wandb
import torch
import gc

from utils.loss import condition_number

from pdb import set_trace as stx


def save_loss_plot(loss_values, results_dir, digit1=None, digit2=None):
    name = '' if digit1 is None or digit2 is None else f'{digit1}_{digit2}'
    plt.figure(figsize=(6, 4))
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve {name}')
    plt.legend()
    output_path = os.path.join(results_dir, f'loss{name}.png')
    plt.savefig(output_path)
    plt.close()

    # Optionally, log the loss plot to WandB
    wandb.log({"loss_plot": wandb.Image(output_path)})


def compute_ig(X1, X2, phi):
    # Flatten X1 and compute condition numbers
    X1_flat =  X1.view(X1.size(0), -1)
    cond_X1 = condition_number(X1_flat.T @ X1_flat)
    try:
        X1_phi = phi(X1)
    except:
        X1_phi = X1_flat @ phi
    cond_A1 = condition_number(X1_phi.T @ X1_phi)

    # Flatten X2 and compute condition numbers
    X2_flat = X2.view(X2.size(0), -1)
    cond_X2 = condition_number(X2_flat.T @ X2_flat)
    try:
        X2_phi = phi(X2)
    except:
        X2_phi = X2_flat @ phi
    cond_A2 = condition_number(X2_phi.T @ X2_phi)

    ratio = torch.exp(cond_A2 - cond_X2 - cond_A1 + cond_X1)

    return ratio

def log_and_save_condition_numbers(X1, X2, X1_immu, X2_immu, results_dir, digit1=None, digit2=None, psd=True):
    # Flatten X1 and compute condition numbers
    X1_flat =  X1.view(X1.size(0), -1)
    log_cond_X1 = condition_number(X1_flat.T @ X1_flat)
    X1_immu_flat =  X1_immu.view(X1_immu.size(0), -1)
    log_cond_A1 = condition_number(X1_immu_flat.T @ X1_immu_flat)

    # Flatten X2 and compute condition numbers
    X2_flat = X2.view(X2.size(0), -1)
    log_cond_X2 = condition_number(X2_flat.T @ X2_flat)
    X2_immu_flat =  X2_immu.view(X2_immu.size(0), -1)
    log_cond_A2 = condition_number(X2_immu_flat.T @ X2_immu_flat)

    ratio = torch.exp(log_cond_A2 - log_cond_X2 - log_cond_A1 + log_cond_X1)

    # Log condition numbers and correlation statistics
    print(f"X1 condition: {torch.exp(log_cond_X1).item()}, A1 condition: {torch.exp(log_cond_A1).item()}")
    print(f"X2 condition: {torch.exp(log_cond_X2).item()}, A2 condition: {torch.exp(log_cond_A2).item()}")
    print(f"Immunization Gap: {ratio}")



    # Save ratios and correlation summary to file
    ratios_file = os.path.join(results_dir, "ratios.txt")
    with open(ratios_file, "a") as f:
        if digit1 is not None and digit2 is not None:
            f.write(f"{digit1} {digit2} {torch.exp(log_cond_A1 - log_cond_X1).item()} {torch.exp(log_cond_A2 - log_cond_X2).item()} {ratio}\n ")
        else:
            f.write(f"{torch.exp(log_cond_A1 - log_cond_X1).item()} {torch.exp(log_cond_A2 - log_cond_X2).item()} {ratio}\n ")
    
    wandb.log({"Ratio 1": torch.exp(log_cond_A1 - log_cond_X1)})
    wandb.log({"Ratio 2": torch.exp(log_cond_A2 - log_cond_X2).item()})
    wandb.log({"IG": ratio})



def log_and_save_avg_condition_numbers(data_module, feature_extractor_ori, feature_extractor_immu, k, n, output_dir, device, train=True):
    
    ratio1 = []
    ratio2 = []
    immunization_gaps = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ratios.txt")

    with torch.no_grad():
        # Compute condition numbers
        for i in range(n):
            # Sample k data points
            (X1, _), (X2, _) = data_module.get_sampled_data(k, train=train)
            X1 = X1.to(device, dtype=torch.double)
            X2 = X2.to(device, dtype=torch.double)

            X1_wo = feature_extractor_ori(X1)
            X2_wo = feature_extractor_ori(X2)
            X1_immu = feature_extractor_immu(X1)
            X2_immu = feature_extractor_immu(X2)

            X1_flat = X1_wo.view(X1_wo.size(0), -1)
            log_cond_X1 = condition_number(X1_flat.T @ X1_flat)
            X1_phi_flat = X1_immu.view(X1_immu.size(0), -1)
            log_cond_A1 = condition_number(X1_phi_flat.T @ X1_phi_flat)

            X2_flat = X2_wo.view(X2_wo.size(0), -1)
            log_cond_X2 = condition_number(X2_flat.T @ X2_flat)
            X2_phi_flat = X2_immu.view(X2_immu.size(0), -1)
            log_cond_A2 = condition_number(X2_phi_flat.T @ X2_phi_flat)

            # Compute the immunization gap
            immunization_gap = torch.exp(log_cond_A2 - log_cond_X2 - log_cond_A1 + log_cond_X1)

            ratio1.append(torch.exp(log_cond_A1 - log_cond_X1).item())
            ratio2.append(torch.exp(log_cond_A2 - log_cond_X2).item())
            immunization_gaps.append(immunization_gap.item())
        # Cleanup tensors manually to free memory
        del X1, X2, X1_wo, X2_wo, X1_immu, X2_immu
        del X1_flat, X1_phi_flat, X2_flat, X2_phi_flat, log_cond_X1, log_cond_A1, log_cond_X2, log_cond_A2, immunization_gap

        # Explicitly release unused memory
        torch.cuda.empty_cache()
        gc.collect()



    # Compute statistics
    ratio1_mean, ratio1_std = torch.tensor(ratio1).mean().item(), torch.tensor(ratio1).std().item()
    ratio2_mean, ratio2_std = torch.tensor(ratio2).mean().item(), torch.tensor(ratio2).std().item()
    gap_mean, gap_std = torch.tensor(immunization_gaps).mean().item(), torch.tensor(immunization_gaps).std().item()

    # Save statistics to file
    with open(output_file, "a") as f:
        f.write(f"{ratio1_mean} {ratio2_mean} {gap_mean}\n")
    with open(output_file + '_std', "a") as f:
        f.write(f"{ratio1_std} {ratio2_std} {gap_std}\n")

def log_accuracy(accuracy, experiment_dir):
    """
    Logs and saves the accuracy of the model evaluation.
    
    Args:
        accuracy (float): Accuracy of the model on the evaluation dataset.
        experiment_dir (str): Directory where logs are stored.
        immunized (bool): Whether this accuracy corresponds to the immunized model.
    """
    # Create log directory if it does not exist
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Define the file name based on whether it is immunized or not
    log_file = "acc.txt"
    log_path = os.path.join(experiment_dir, log_file)
    
    # Log accuracy to file
    with open(log_path, "w") as log:
        log.write(f"Accuracy: {accuracy:.4f}\n")
    
    print(f"Logged accuracy: {accuracy:.4f} to {log_path}")