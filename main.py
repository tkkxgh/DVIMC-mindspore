from datasets import build_dataset
import mindspore as ms
from mindspore import nn, grad, Tensor, ops, value_and_grad
from mindspore.experimental import optim
from dvimc_model import DVIMC_model
from dvimc_loss import DVIMC_loss
from run_epoch import train, test
import numpy as np
import random
import argparse
from sklearn.cluster import KMeans


def initialization(model, cmv_data, sv_loaders, args):
    print('Initializing......')
    model.set_train(True)
    criterion = nn.MSELoss()

    def initialization_forward(sv_data, view):
        _, sv_rec = model.vs_encode(sv_data, view)
        vs_rec_loss = criterion(sv_rec, sv_data)
        return vs_rec_loss

    for v in range(args.num_views):

        networks_parameters = list(filter(lambda p: f'view_{v}' in p.name and 'var' not in p.name, model.trainable_params()))
        optimizer = optim.Adam(networks_parameters)
        for e in range(1, args.initialization_epochs + 1):
            for (xv,) in sv_loaders[v]:
                grad_fn = value_and_grad(initialization_forward, None, optimizer.parameters, has_aux=False)
                rec_loss, params_gradient = grad_fn(xv, v)
                optimizer(params_gradient)

    model.set_train(False)
    fit_data = [Tensor(csv_data, dtype=ms.float32) for csv_data in cmv_data]
    latent_representations = []
    for v in range(args.num_views):
        latent, _ = model.vs_encode(fit_data[v], v)
        latent_representations.append(latent)
    fused_latent_representations = sum(latent_representations) / len(latent_representations)
    kmeans = KMeans(n_clusters=args.class_num, n_init=10)
    kmeans.fit(fused_latent_representations.asnumpy())
    model.prior_mu.set_data(Tensor(kmeans.cluster_centers_, dtype=ms.float32))


def main(args):
    print(f"Dataset : {args.dataset_name:>10} Missing rate : {args.missing_rate}")
    eval_record = {"ACC": [], "NMI": [], "PUR": [], "ARI": []}
    random.seed(2)
    np.random.seed(3)
    ms.set_seed(4)
    cmv_data, imv_loader, sv_loaders = build_dataset(args)
    model = DVIMC_model(args)
    networks_parameters = list(filter(lambda p: 'encoders' in p.name or 'decoders' in p.name, model.trainable_params()))
    prior_parameters = list(filter(lambda p: 'prior' in p.name, model.trainable_params()))
    group_params = [{'params': networks_parameters, 'lr': args.learning_rate},
                    {'params': prior_parameters, 'lr': args.prior_learning_rate}]
    optimizer = optim.Adam(group_params)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    # milestone = list(range(args.lr_decay_step, args.train_epochs + 1, args.lr_decay_step))
    # learning_rate = [args.learning_rate * pow(args.lr_decay_gamma, step) for step in range(len(milestone))]
    # prior_learning_rate = [args.prior_learning_rate * pow(args.lr_decay_gamma, step) for step in range(len(milestone))]
    # lr = nn.piecewise_constant_lr(milestone, learning_rate)
    # prior_lr = nn.piecewise_constant_lr(milestone, prior_learning_rate)

    # optimizer = nn.Adam([{'params': networks_parameters, 'lr': lr},
    #                 {'params': prior_parameters, 'lr': prior_lr}])
    initialization(model, cmv_data, sv_loaders, args)
    dvimc_loss = DVIMC_loss(args)

    print('training...')
    for epoch in range(1, args.train_epochs + 1):
        epoch_loss = train(model, dvimc_loss, optimizer, scheduler, imv_loader)
        if epoch % args.log_interval == 0:
            acc, nmi, ari, pur = test(model, imv_loader)
            print(f'Epoch {epoch:>3}/{args.train_epochs} Loss : {epoch_loss:.2f} '
                  f'ACC : {acc * 100:.2f} NMI: {nmi * 100:.2f} ARI: {ari * 100:.2f} PUR: {pur * 100:.2f}')

    final_results = test(model, imv_loader)
    eval_record["ACC"].append(final_results[0])
    eval_record["NMI"].append(final_results[1])
    eval_record["ARI"].append(final_results[2])
    eval_record["PUR"].append(final_results[3])
    print(f'Average Results : ACC {np.mean(eval_record["ACC"]) * 100:.2f} NMI {np.mean(eval_record["NMI"]) * 100:.2f} '
          f'ARI {np.mean(eval_record["ARI"]) * 100:.2f} PUR {np.mean(eval_record["PUR"]) * 100:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', type=int, default=300, help='training epochs')
    parser.add_argument('--initialization_epochs', type=int, default=200, help='initialization epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--prior_learning_rate', type=float, default=0.05, help='initial mixture-of-gaussian learning rate')
    parser.add_argument('--z_dim', type=int, default=10, help='latent dimensions')
    parser.add_argument('--lr_decay_step', type=float, default=10, help='StepLr_Step_size')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.9, help='StepLr_Gamma')

    parser.add_argument('--dataset', type=int, default=0, help='0:Caltech7-5v, 1:Scene-15, 2:Multi-Fashion, 3:NoisyMNIST')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--runs_num', type=int, default=10)
    parser.add_argument('--missing_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=5)

    args = parser.parse_args()
    ms.set_context(device_id=0, device_target="GPU")
    args.dataset_dir_base = "./npz_data/"

    if args.dataset == 0:
        args.dataset_name = 'Caltech7-5V'
        args.alpha = 5
        args.likelihood = 'Gaussian'
    elif args.dataset == 1:
        args.dataset_name = 'Scene-15'
        args.alpha = 20
        args.likelihood = 'Gaussian'
    elif args.dataset == 2:
        args.dataset_name = 'Multi-Fashion'
        args.alpha = 10
        args.likelihood = 'Bernoulli'
    else:
        args.dataset_name = 'NoisyMNIST'
        args.alpha = 10
        args.likelihood = 'Bernoulli'
        args.batch_size = 512

    for missing_rate in [0.1, 0.3, 0.5, 0.7]:
        args.missing_rate = missing_rate
        main(args)
