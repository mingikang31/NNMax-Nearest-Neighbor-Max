import torch
import torch.nn as nn
import torch.nn.functional as F

class Softmax(nn.Module):
    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Computes softmax along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int, optional): Dimension along which to compute softmax. If None, uses the dimension specified in __init__. Defaults to None.

        Returns:
            torch.Tensor: Softmax output.
        """
        if self.dim is None:
            dim = -1  # Apply softmax over the last dimension if dim is not specified
        else:
            dim = self.dim

        # Subtract the maximum value for numerical stability
        x = x - torch.max(x, dim=dim, keepdim=True)[0]

        # Compute exponentials
        exp_x = torch.exp(x)

        # Compute softmax
        softmax_x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

        return softmax_x
    
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(x)
print(x.shape)
print()

print("Scratch softmax")
softmax = Softmax(dim=-1)
output = softmax(x)
print(output)
print(output.shape)

print()
print("Torch softmax")
output = F.softmax(x, dim=-1)
print(output)
print(output.shape)