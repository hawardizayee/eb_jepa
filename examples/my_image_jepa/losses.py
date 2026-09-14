import torch 
import torch.nn as nn 
import torch.nn.functional as F 



class HingeVarLoss(torch.nn.Module):
    def __init__(
        self,
        var_margin: float = 1.0
    ):
        """
        Encourages each feature to maintain at least a minimum variance of one. 
        Features with var below the margin incur a penalty of (var_margin = var).
        Args:
            var_margin (float, default=1.0)
                Minimum desired variance per feature.
        """
        super().__init__()
        self.var_margin = var_margin

    def forward(self, x : torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            var_loss: Scalar tensor loss on Variance.
        """
        var = x.var(dim=0)
        var_loss = torch.mean(F.relu(self.var_margin - var))
        return var_loss 

class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelations. 

        Normalizes by D * (D - 1) where D is feature dimensionality.
        """
        super().__init__()
    
    def off_diagonal(self,x): 
        n,m = x.shape 
        assert n == m 
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x = x - x.mean(dim=0,keepdim=True)
        cov = (x.T @ x) / (batch_size -1) # [D,D]
        # calculate off-diagonal loss 
        cov_loss = self.off_diagonal(cov).pow(2).mean()

        return cov_loss

class VICRegLoss(nn.Module):
    """VICReg loss combining invariance, variance, and covariance terms."""

    def __init__(self,var_coeff=1.0,cov_coeff=1.0):
        super().__init__()
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.var_loss_fn = HingeVarLoss(var_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, z1, z2):
        """compute VICReg loss.
        
        Agrs:
            z1: [B,D] - First projection tensor 
            z2: [B,D] - Second projection tensor 

        Returns: 
            dict with keys: loss,invariance_loss, var_loss, cov_loss
        """ 

        # Invarinace loss (similarity)
        sim_loss = F.mse_loss(z1,z2)

        # Variance loss (applied to both views and summed)
        var_loss = self.var_loss_fn(z1) + self.var_loss_fn(z2)

        # Covariance loss (applied to both views and summed)
        cov_loss = self.cov_loss_fn(z1) + self.cov_loss_fn(z2)

        total_loss = sim_loss + self.var_coeff * var_loss + self.cov_coeff * cov_loss

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss
        }
          

######################################################
# BCS (Batched Characteristic Slicing) loss for SIReg

def all_reduce(x,op):
    """All-reduce operation for distributed training."""
    import torch.distributed as dist 

    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x,op=op)
        return x 
    else:
        return x 
    

def epps_pulley(x, t_min=-3, t_max=3, n_points=10):
    """Epps-Pulley test statistic for Guassianity."""
    # integration points 
    t = torch.linspace(t_min,t_max,n_points,device=x.device)
    # theoritical CF for N(0, 1)
    exp_f = torch.exp(-0.5 * t**2)
    # ECF 
    x_t = x.unsqueeze(2) * t # (N,M,T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    # weighted L2 distance 
    err = exp_f * (ecf - exp_f).abs()**2 
    T = torch.trapz(err, t, dim=1)
    return T 


class BCS(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(self,num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0 
        self.lmbd = lmbd

    def forward(self,z1,z2):
        with torch.no_grad():
            dev = z1.device 
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)

        view1 = z1 @ A 
        view2 = z2 @ A 

        self.step += 1
        bcs = (epps_pulley(view1).mean() + epps_pulley(view2).mean()) / 2 
        invariance_loss = F.mse_loss(z1,z2).mean()
        total_loss = invariance_loss + self.lmbd * bcs 
        return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}