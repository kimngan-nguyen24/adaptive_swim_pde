from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class AdaptiveSolver:
    optimizer: str = "Adam"
    loss_function: str = "MSE"
    learning_rate: float = 3e-4 # old: 1e-3 doesn't work well
    regularization_scale: float = 1e-6
    max_epochs: int = 1000

    tolerance: float = 1e-6
    batch_size: int = 64
    shuffle: bool = True
    random_seed: int = 42

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        pass

    def compute_a_params(
            self, 
            model: nn.Module, 
            x_points: torch.Tensor,
            y_points: torch.Tensor,
            w: torch.Tensor,
            b: torch.Tensor,
            ) -> torch.Tensor:
        """
        Compute adaptive parameters for the model.

        Args:
            model (nn.Module): neural network model
            x_points (torch.Tensor): x interpolation points (N, k, d)
            y_points (torch.Tensor): y interpolation points (N, k, 1)
            w (torch.Tensor): weights (N, d)
            b (torch.Tensor): bias (N, 1)
        """
        x_1d = torch.matmul(x_points, w.unsqueeze(-1)) + b.unsqueeze(1)
        # (N, k, 1) + (N, 1, 1) = (N, k, 1)
        
        if isinstance(self.loss_function, str):
            match self.loss_function:
                case "MSE":
                    self.loss_function = nn.MSELoss()
                case "cosine":
                    self.loss_function = self.cosine_loss
                case _:
                    raise ValueError(
                        f"Loss function '{self.loss_function}' is not supported."
                    )
        
        if self.optimizer == "Adam":
            return self.adam_optimize(model, x_1d.squeeze(-1), y_points.squeeze(-1))
    

    def cosine_loss(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
        """
        Computes the cosine loss between outputs and targets.

        Args:
            outputs (torch.Tensor): model outputs (N, k)
            targets (torch.Tensor): target values (N, k)
        """
        norm_outputs = torch.norm(outputs, dim=1)
        norm_targets = torch.norm(targets, dim=1)
        dot_product = torch.sum(outputs * targets, dim=1)

        loss = 1 - (dot_product / (norm_outputs * norm_targets + 1e-8))
        return torch.mean(loss)


    def adam_optimize(
            self, 
            model: nn.Module,
            x: torch.Tensor,
            y: torch.Tensor,
            ) -> torch.Tensor:
        """
        Trains the model using the Adam optimizer.

        Args:
            model (nn.Module): neural network model
            x (torch.Tensor): input data (N, k) after linear transformation and before activation
            y (torch.Tensor): target data (N, k)
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        x = x.to(self.device)
        y = y.to(self.device)
        model = model.to(self.device)

        best_loss = float('inf')
        best_params = None
        best_epoch = 0
        patience = self.max_epochs // 5

        for epoch in range(self.max_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x)
            loss = self.loss_function(outputs, y) # + self.regularization_scale * torch.sum(torch.square(model.a_params))
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss - self.tolerance:
                best_loss = current_loss
                best_params = [param.clone().detach().cpu() for param in model.parameters()][0]
                best_epoch = epoch
            elif epoch - best_epoch >= patience:
                # Early stopping
                print(
                    f"Early stopping at epoch {epoch}, best was at {best_epoch} with loss {best_loss:.6f}"
                )
                break

        return best_params