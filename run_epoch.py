from mindspore import value_and_grad, ops, Tensor, nn
from evaluate import evaluate
import numpy as np
import mindspore as ms

from mindspore.ops import functional as F
from mindspore.ops import composite as CC


def train(model, dvimc_loss, optimizer, scheduler, imv_loader):
    model.set_train(True)

    def forward_fn(imv_data, mask):
        _, vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample = model(imv_data, mask)
        loss = dvimc_loss(vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample, model.prior_weight, model.prior_mu,
                          model.prior_var, imv_data, mask)
        return loss

    epoch_loss_record = []
    for batch_idx, item in enumerate(imv_loader):
        data_list = item[:-2]
        mask_matrix = item[-2]
        grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
        total_loss, params_gradient = grad_fn(data_list, mask_matrix)
        epoch_loss_record.append(total_loss.asnumpy())
        # params_gradient = ops.clip_by_value(params_gradient, -1, 1)
        optimizer(params_gradient)
        model.prior_weight.set_data(model.prior_weight.value() / ops.sum(model.prior_weight.value(), keepdim=True))
    scheduler.step()
    epoch_loss = sum(epoch_loss_record) / len(epoch_loss_record)
    return epoch_loss


def log_gaussian(x, mu, var):
    return -0.5 * (ops.log(Tensor(2.0 * np.pi, dtype=ms.float32)) + ops.log(var) + ops.pow(x - mu, 2) / var)


def mog_predict(mu, mog_weight, mog_mu, mog_var):
    log_pz_c = ops.sum(log_gaussian(mu.unsqueeze(1), mog_mu.unsqueeze(0), mog_var.unsqueeze(0)), dim=-1)
    log_pc = ops.log(mog_weight.unsqueeze(0))
    log_pc_z = log_pc + log_pz_c
    pc_z = ops.exp(log_pc_z) + 1e-10
    normalized_pc_z = pc_z / ops.sum(pc_z, dim=1, keepdim=True)
    return normalized_pc_z


def test(model, imv_loader):
    model.set_train(False)
    c_assignment = []
    true_labels = []
    for batch_idx, item in enumerate(imv_loader):
        imv_data = item[:-2]
        mask = item[-2]
        labels = item[-1]
        _, _, _, aggregated_mu, _, _, _ = model(imv_data, mask)
        mog_weight, mog_mu, mog_var = model.prior_weight.value(), model.prior_mu.value(), model.prior_var.value()
        c_assignment.append(mog_predict(aggregated_mu, mog_weight, mog_mu, mog_var))
        true_labels.append(labels)
    true_labels = ops.cat(true_labels, axis=0).asnumpy()
    c_assignment = ops.cat(c_assignment, axis=0)
    predict = ops.argmax(c_assignment, dim=1).asnumpy()
    acc, nmi, ari, pur = evaluate(true_labels, predict)
    return acc, nmi, ari, pur
