import torch
import torch.nn.functional as F
import torch.nn as nn
from pdb import set_trace as stx

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def reconstruction_loss(X, y, phi, w):
    X_phi = phi(X)
    reconstruction_error = torch.norm(y.float() - X_phi @ w, 2) ** 2 / y.size(0)
    return reconstruction_error


def condition_number(S, lambda_diag=1e-6, return_eigvals=False):

    S = S.to(dtype=torch.double)
    S = S + torch.eye(S.shape[0], dtype=S.dtype, device=S.device) * lambda_diag
    kappa = torch.linalg.eigvalsh(S)

    sigma_max = torch.max(kappa)
    sigma_min = torch.min(kappa[kappa > lambda_diag])

    if return_eigvals:
        return sigma_min, sigma_max, torch.log(sigma_max) - torch.log(sigma_min)
    
    log_condition_number = torch.log(sigma_max) - torch.log(sigma_min)

    return log_condition_number

def kappa_loss(X1, X2, phi, lambda_cond=1e-5):
    X1_phi = phi(X1)
    A1_phi = X1_phi.T @ X1_phi
    kappa1 = condition_number(A1_phi)

    X2_phi = phi(X2)
    A2_phi = X2_phi.T @ X2_phi
    kappa2 = condition_number(A2_phi)

    return 1 / torch.log(kappa2) + lambda_cond * torch.log(kappa1)



def inner_optimization(X1, y1, w1, phi, num_inner_steps=30, lr=1e-4):
    # Perform a differentiable SGD update using a closure
    for _ in range(num_inner_steps):
        loss_inner = reconstruction_loss(X1, y1, phi, w1)
        
        grads = torch.autograd.grad(loss_inner, phi.parameters(), create_graph=True)
        
        with torch.no_grad():
            for param, grad in zip(phi.parameters(), grads):
                param -= lr * grad

    return phi, loss_inner.item()

def outer_loss(X2, phi_star):
    X2_phi = phi_star(X2)
    A2_phi = X2_phi.T @ X2_phi
    kappa2 = condition_number(A2_phi)

    return 1 / torch.log(kappa2)

def outer_optimization(X1, y1, X2, w1, phi, num_outer_steps=1000, lr=1e-3):
    optimizer_outer = torch.optim.SGD([w1], lr=lr)
    
    for _ in range(num_outer_steps):
        optimizer_outer.zero_grad()
        
        phi_star = inner_optimization(X1, y1, w1, phi)
        
        outer_loss = kappa_loss(X1, X2, phi_star)
        
        outer_loss.backward()
        optimizer_outer.step()


    return w1, phi_star


def S1_gradients(S):
    """
    Compute the gradients based on SVD decomposition in PyTorch.

    Args:
        S (torch.Tensor): Input matrix of shape (n, n), assumed to be square and symmetric.

    Returns:
        torch.Tensor: Computed gradient matrix.
        float: Condition number (kappa) of the matrix.
    """
    # Ensure the input is a float tensor
    S = S.to(dtype=torch.float64)

    # Perform SVD decomposition
    U, sigmas, Vh = torch.linalg.svd(S)
    V = Vh.T

    # Compute the condition number
    kappa = sigmas[0].item() / sigmas[-1].item()

    # Extract the first singular vector pair
    u1 = U[:, 0].unsqueeze(1)  # First left singular vector (column vector)
    v1 = V[:, 0].unsqueeze(1)  # First right singular vector (column vector)

    # Compute the gradient of the largest singular value
    grad_sigma = sigmas[0] * (u1 @ v1.T)

    # Compute the Frobenius gradient
    grad_F = S / S.shape[0]

    # Compute the final gradient
    grad = grad_sigma - grad_F

    return grad, kappa


def S2_gradients(S):
    """
    Compute the gradients based on the smallest singular value using SVD decomposition in PyTorch.

    Args:
        S (torch.Tensor): Input matrix of shape (n, n), assumed to be square and symmetric.

    Returns:
        torch.Tensor: Computed gradient matrix.
        float: Condition number (kappa) of the matrix.
    """
    # Ensure the input is a float tensor
    S = S.to(dtype=torch.float64)

    # Perform SVD decomposition
    U, sigmas, Vh = torch.linalg.svd(S)
    V = Vh.T

    # Compute the condition number (kappa)
    kappa = sigmas[0].item() / sigmas[-1].item()

    # Extract the last singular vector pair (corresponding to the smallest singular value)
    uk = U[:, -1].unsqueeze(1)  # Last left singular vector (column vector)
    vk = V[:, -1].unsqueeze(1)  # Last right singular vector (column vector)

    # Compute the gradient of the smallest singular value
    grad_sigma = sigmas[-1] * (uk @ vk.T)

    # Compute the Frobenius gradient
    grad_F = S / S.shape[0]

    # Compute the normalization factor
    fro_norm = torch.norm(S, p="fro")
    normalization_factor = (fro_norm / (2 * S.shape[0]) + sigmas[-1]**2 / 2)**2

    # Compute the final gradient
    grad = (grad_sigma - grad_F) / normalization_factor

    return grad, kappa