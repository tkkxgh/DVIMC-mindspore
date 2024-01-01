from typing import Optional

from mindspore import nn, ops, Tensor
import mindspore as ms
import numpy as np


def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None):
    max_val = ops.maximum(-input, 0)

    if pos_weight is not None:
        log_weight = ((pos_weight - 1) * target) + 1
        loss = (1 - target) * input
        loss_1 = ops.log(ops.exp(-max_val) + ops.exp(-input - max_val)) + max_val
        loss += log_weight * loss_1
    else:
        loss = (1 - target) * input
        loss += max_val
        loss += ops.log(ops.exp(-max_val) + ops.exp(-input - max_val))

    if weight is not None:
        output = loss * weight
    else:
        output = loss

    if reduction == "mean":
        return ops.reduce_mean(output)
    elif reduction == "sum":
        return ops.reduce_sum(output)
    else:
        return output


class BCEWithLogitsLoss(nn.Cell):
    reduction_list = ['sum', 'mean', 'none']

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None):
        super().__init__()
        if reduction not in self.reduction_list:
            raise ValueError(f'Unsupported reduction {reduction}')
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def construct(self, input, target):
        return binary_cross_entropy_with_logits(input, target, self.weight, self.reduction, self.pos_weight)


class DVIMC_loss(nn.Cell):
    def __init__(self, args):
        super().__init__()
        self.likelihood = args.likelihood
        self.alpha = args.alpha
        if self.likelihood == 'Bernoulli':
            self.rec_fn = BCEWithLogitsLoss(reduction='none')
        else:
            self.rec_fn = nn.MSELoss(reduction='none')

    def log_gaussian(self, x, mu, var):
        return -0.5 * (ops.log(Tensor(2.0 * np.pi, dtype=ms.float32)) + ops.log(var) + ops.pow(x - mu, 2) / var)

    def gaussian_kl(self, q_mu, q_var, p_mu, p_var):
        return - 0.5 * (ops.log(q_var / p_var) - q_var / p_var - ops.pow(q_mu - p_mu, 2) / p_var + 1)

    def vade_trick(self, mc_sample, prior_weight, prior_mu, prior_var):
        log_pz_c = ops.sum(self.log_gaussian(mc_sample.unsqueeze(1), prior_mu.unsqueeze(0), prior_var.unsqueeze(0)), dim=-1)
        log_pc = ops.log(prior_weight.unsqueeze(0))
        log_pc_z = log_pc + log_pz_c
        pc_z = ops.exp(log_pc_z) + 1e-10
        normalized_pc_z = pc_z / ops.sum(pc_z, dim=1, keepdim=True)
        return normalized_pc_z

    def kl_term(self, z_mu, z_var, qc_x, prior_weight, prior_mu, prior_var):
        z_kl_div = ops.sum(qc_x * ops.sum(self.gaussian_kl(z_mu.unsqueeze(1), z_var.unsqueeze(1), prior_mu.unsqueeze(0), prior_var.unsqueeze(0)),
                                          dim=-1), dim=1)
        z_kl_div_mean = ops.mean(z_kl_div)

        c_kl_div = ops.sum(qc_x * ops.log(qc_x / prior_weight.unsqueeze(0)), dim=1)
        c_kl_div_mean = ops.mean(c_kl_div)
        return z_kl_div_mean + c_kl_div_mean

    def coherence_term(self, vs_mus, vs_vars, aggregated_mu, aggregated_var, mask=None):
        mv_coherence_loss = []
        norm = ops.sum(mask, dim=1)
        for v in range(len(vs_mus)):
            sv_coherence_loss = ops.sum(self.gaussian_kl(aggregated_mu, aggregated_var, vs_mus[v], vs_vars[v]), dim=1)
            exist_loss = ops.mul(sv_coherence_loss, mask[:, v])
            mv_coherence_loss.append(exist_loss)
        coherence_loss = ops.mean(sum(mv_coherence_loss) / norm)
        return coherence_loss

    def construct(self, vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample, prior_weight, prior_mu, prior_var, imv_data,
                  mask):
        # z_sample, vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample = self.model(imv_data, mask)
        qc_x = self.vade_trick(vade_z_sample, prior_weight, prior_mu, prior_var)
        kl_loss = self.kl_term(aggregated_mu, aggregated_var, qc_x, prior_weight, prior_mu, prior_var)
        coherence_loss = self.coherence_term(vs_mus, vs_vars, aggregated_mu, aggregated_var, mask)

        mv_rec_loss = []
        for v in range(len(imv_data)):
            rec_loss = ops.sum(self.rec_fn(xr_list[v], imv_data[v]), dim=1)
            exist_rec = rec_loss * mask[:, v]
            sv_rec_loss = ops.mean(exist_rec)
            mv_rec_loss.append(sv_rec_loss)
        rec_loss = sum(mv_rec_loss)
        return rec_loss + kl_loss + self.alpha * coherence_loss
