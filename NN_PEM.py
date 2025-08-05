import torch
import torch.nn as nn
import torch.nn.functional as F

# === Define Neural Network ===
class BoundingBoxErrorNet(nn.Module):
    def __init__(self, in_features=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 14)  # 4 means + 10 Cholesky entries
        )

    def forward(self, x):
        out = self.net(x)

        mu = out[:, :4]  # mean
        l_params = out[:, 4:]  # 10 params for Cholesky

        L = torch.zeros((x.size(0), 4, 4), device=x.device)
        idx = 0
        for i in range(4):
            for j in range(i + 1):
                if i == j:
                    L[:, i, j] = F.softplus(l_params[:, idx]) + 1e-3  # Ensure positivity - similar to ReLU: postprocess activation
                else:
                    L[:, i, j] = l_params[:, idx]
                idx += 1

        return mu, L


def multivariate_gaussian_nll(y, mu, L):
    """
    Computes the negative log-likelihood of y under N(mu, LL^T)
    y: (B, 4) - ground truth
    mu: (B, 4) - predicted mean
    L: (B, 4, 4) - Cholesky factor
    """
    B = y.size(0)
    diff = (y - mu).unsqueeze(-1)  # (B, 4, 1)

    # Solve L * z = diff --> z = L^{-1} * diff
    Linv_diff = torch.linalg.solve_triangular(L, diff, upper=False)  # (B, 4, 1)
    mahalanobis = torch.sum(Linv_diff ** 2, dim=(1, 2))  # (B,)

    # log(det(Sigma)) = 2 * sum(log(diag(L)))
    log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2)), dim=1)  # (B,)

    log_likelihood = -0.5 * (mahalanobis + log_det + 4 * torch.log(torch.tensor(2 * torch.pi)))
    return -log_likelihood.mean()


if __name__ == "__main__":
    model = BoundingBoxErrorNet()
    dummy_input = torch.randn(8, 5, 64, 64)  # (batch, channels, H, W)
    dummy_target = torch.randn(8, 4)  # Ground truth bbox error

    mu_pred, L_pred = model(dummy_input)
    loss = multivariate_gaussian_nll(dummy_target, mu_pred, L_pred)
    print(f"Loss: {loss.item():.4f}")
