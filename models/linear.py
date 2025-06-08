import torch
from torch import nn
from torch.autograd import Function

# Linear model with preconditioner solved via ridge regression
class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, lambda_reg=1.0, use_precond=True):
        # Save necessary tensors and the regularization parameter for backward
        ctx.save_for_backward(input, weight, bias)
        ctx.lambda_reg = lambda_reg
        ctx.use_precond = use_precond

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        lambda_reg = ctx.lambda_reg
        use_precond = ctx.use_precond

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        
        if ctx.needs_input_grad[1]:
            base_grad_weight = grad_output.t().mm(input)
            if use_precond:
                # Ridge regression solution: (XtX + lambda * I) w = XtX * base_grad_weight
                XtX = input.t().mm(input)
                lambda_eye = lambda_reg * torch.eye(XtX.size(0), device=XtX.device, dtype=XtX.dtype)
                XtX_reg = XtX + lambda_eye
                grad_weight = torch.linalg.solve(XtX_reg, base_grad_weight)
            else:
                grad_weight = base_grad_weight

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None


class Linear(nn.Linear):
    def forward(self, input, lambda_reg=1.0, use_precond=False):
        # Dynamically decide whether to use the preconditioner
        return LinearFunction.apply(input, self.weight, self.bias, lambda_reg, use_precond)