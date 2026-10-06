"""
Uncertainty Module - Probabilistic representation, reparameterization, KL divergence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class UncertaintyBlock(nn.Module):
    """
    Uncertainty Block (GPE - Graph-based Probabilistic Encoder).
    
    Models features as Gaussian distribution N(mu, sigma^2) with:
    - Reparameterization trick for sampling
    - KL divergence regularization towards N(0, I)
    - Attention gating on mu and sigma
    """
    
    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_attention = use_attention
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        
        # Mean and variance heads
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),  # Bound to (0, 1)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),  # Bound to (0, 1)
        )
        
        # Attention gating
        if use_attention:
            self.attention_mu = SimpleAttention(out_dim)
            self.attention_var = SimpleAttention(out_dim)
        else:
            self.attention_mu = nn.Identity()
            self.attention_var = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, in_dim] - input features
            
        Returns:
            sample: [B, out_dim] - reparameterized sample
            kl_loss: scalar - KL divergence loss
        """
        encoded = self.encoder(x)  # [B, hidden_dim]
        
        mu = self.fc_mu(encoded)      # [B, out_dim]
        var = self.fc_var(encoded)    # [B, out_dim]
        
        # Attention gating
        mu_attn = self.attention_mu(mu)
        var_attn = self.attention_var(var)
        
        # Reparameterization
        sample = self.reparameterize(mu_attn, var_attn)
        
        # KL loss
        kl_loss = self.kl_loss(mu_attn, var_attn)
        
        return sample, kl_loss
    
    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * sigma
        
        Note: var is treated as VARIANCE (not log-variance)
        sigma = sqrt(var)
        """
        std = var.sqrt()  # Treat var as variance
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_loss(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between N(mu, var) and N(0, I).
        
        Formula: KL = -0.5 * sum(1 + log(var) - mu^2 - var)
        where var is the VARIANCE (not log-variance).
        
        Note: This uses the correct formula for variance parameterization.
        The original MUAD code had an inconsistency where var was used
        as variance in reparameterize but as log-variance in kl_loss.
        This implementation uses variance consistently.
        """
        # KL(N(mu, var) || N(0, I)) = 0.5 * (mu^2 + var - 1 - log(var))
        kl = 0.5 * (mu ** 2 + var - 1 - var.log())
        return kl.mean()


class SimpleAttention(nn.Module):
    """
    Simple attention gating mechanism.
    x -> x * sigmoid(Linear(x))
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class VariationalUncertaintyBlock(nn.Module):
    """
    Improved Uncertainty Block with explicit log-variance parameterization.
    Avoids the variance/log-variance inconsistency in the original code.
    """
    
    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        
        # Mean head (unbounded)
        self.fc_mu = nn.Linear(hidden_dim, out_dim)
        
        # Log-variance head (unbounded, then exp gives variance)
        self.fc_logvar = nn.Linear(hidden_dim, out_dim)
        
        # Attention gating
        if use_attention:
            self.attention_mu = SimpleAttention(out_dim)
            self.attention_var = SimpleAttention(out_dim)
        else:
            self.attention_mu = nn.Identity()
            self.attention_var = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sample: [B, out_dim]
            kl_loss: scalar
        """
        encoded = self.encoder(x)
        
        mu = self.fc_mu(encoded)           # [B, out_dim]
        logvar = self.fc_logvar(encoded)   # [B, out_dim]
        
        # Attention gating
        mu_attn = self.attention_mu(mu)
        logvar_attn = self.attention_var(logvar)
        
        # Reparameterization
        sample = self.reparameterize(mu_attn, logvar_attn)
        
        # KL loss
        kl_loss = self.kl_loss(mu_attn, logvar_attn)
        
        return sample, kl_loss
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + eps * sqrt(exp(logvar))"""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL(N(mu, exp(logvar)) || N(0, I)) 
        = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
        """
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)
        return kl.mean()


class MonteCarloUncertainty(nn.Module):
    """
    Monte Carlo dropout-based uncertainty estimation.
    Runs multiple forward passes with dropout enabled at eval time.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_samples: int = 5,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        
        # Add dropout to base model if not present
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout layers in eval mode."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_prediction: [B, ...] - mean over samples
            uncertainty: [B, ...] - variance over samples
        """
        predictions = []
        
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = self.base_model(*args, **kwargs)
                if isinstance(pred, dict):
                    pred = pred.get("MMlogit", pred.get("y_pred", list(pred.values())[0]))
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_samples, B, ...]
        
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        
        return mean_pred, uncertainty


class EnsembleUncertainty(nn.Module):
    """
    Ensemble-based uncertainty using multiple model copies.
    """
    
    def __init__(
        self,
        model_fn,
        num_models: int = 5,
        **model_kwargs
    ):
        super().__init__()
        self.models = nn.ModuleList([
            model_fn(**model_kwargs) for _ in range(num_models)
        ])
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_prediction: [B, ...]
            uncertainty: [B, ...]
        """
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(*args, **kwargs)
                if isinstance(pred, dict):
                    pred = pred.get("MMlogit", pred.get("y_pred", list(pred.values())[0]))
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, B, ...]
        
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        
        return mean_pred, uncertainty


def compute_predictive_uncertainty(
    logits: torch.Tensor,  # [B, num_classes] or [num_samples, B, num_classes]
    num_classes: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute predictive uncertainty from logits.
    
    Returns:
        entropy: [B] - predictive entropy
        mutual_info: [B] - mutual information (epistemic uncertainty)
    """
    if logits.dim() == 3:
        # [num_samples, B, C]
        probs = F.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)  # [B, C]
        
        # Predictive entropy
        entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=-1)
        
        # Expected entropy
        sample_entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # [num_samples, B]
        expected_entropy = sample_entropy.mean(dim=0)
        
        # Mutual information (epistemic uncertainty)
        mutual_info = entropy - expected_entropy
        
    else:
        # [B, C] - single forward pass
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        mutual_info = torch.zeros_like(entropy)
    
    return entropy, mutual_info