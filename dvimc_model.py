from mindspore import Parameter, nn, ops, jit
import mindspore.nn.probability.distribution as msd
import mindspore as ms


class view_specific_encoder(nn.Cell):
    def __init__(self, view_dim, latent_dim):
        super().__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.encoder = nn.SequentialCell(nn.Dense(self.x_dim, 500),
                                         nn.ReLU(),
                                         nn.Dense(500, 500),
                                         nn.ReLU(),
                                         nn.Dense(500, 2000),
                                         nn.ReLU()
                                         )
        self.z_mu = nn.Dense(2000, self.z_dim)
        self.z_var = nn.Dense(2000, self.z_dim)

    def construct(self, x):
        hidden_feature = self.encoder(x)
        zv_mu = self.z_mu(hidden_feature)
        softplus_fn = ops.Softplus()
        zv_var = softplus_fn(self.z_var(hidden_feature))
        return zv_mu, zv_var


class view_specific_decoder(nn.Cell):
    def __init__(self, view_dim, latent_dim):
        super().__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.decoder = nn.SequentialCell(nn.Dense(self.z_dim, 2000),
                                         nn.ReLU(),
                                         nn.Dense(2000, 500),
                                         nn.ReLU(),
                                         nn.Dense(500, 500),
                                         nn.ReLU(),
                                         nn.Dense(500, self.x_dim)
                                         )

    def construct(self, z):
        xr = self.decoder(z)
        return xr


class DVIMC_model(nn.Cell):
    def __init__(self, args):
        super().__init__()
        self.x_dim_list = args.multiview_dims
        self.k = args.class_num
        self.z_dim = args.z_dim
        self.num_views = args.num_views
        self.normal_dist = msd.Normal(0.0, 1.0)

        self.prior_weight = Parameter(ops.full((self.k,), 1 / self.k, dtype=ms.float32), requires_grad=True)
        self.prior_mu = Parameter(ops.full((self.k, self.z_dim), 0.0, dtype=ms.float32), requires_grad=True)
        self.prior_var = Parameter(ops.full((self.k, self.z_dim), 1.0, dtype=ms.float32), requires_grad=True)
        self.encoders = nn.CellDict({f'view_{v}': view_specific_encoder(self.x_dim_list[v], self.z_dim) for v in range(args.num_views)})
        self.decoders = nn.CellDict({f'view_{v}': view_specific_decoder(self.x_dim_list[v], self.z_dim) for v in range(args.num_views)})

    def inference_z(self, imv_data, mask=None):
        vs_mus, vs_vars = [], []
        for v in range(self.num_views):
            vs_mu, vs_var = self.encoders[f'view_{v}'](imv_data[v])
            vs_mus.append(vs_mu)
            vs_vars.append(vs_var)
        mu = ops.stack(vs_mus)
        var = ops.stack(vs_vars)
        imv_mask = ops.swapaxes(mask, 0, 1)
        imv_mask = imv_mask.unsqueeze(-1)
        exist_mu = ops.mul(mu, imv_mask)
        T = 1. / var
        exist_T = ops.mul(T, imv_mask)
        aggregated_T = ops.sum(exist_T, dim=0)
        aggregated_var = 1. / aggregated_T
        aggregated_mu_numerator = ops.mul(exist_mu, exist_T)
        aggregated_mu = ops.sum(aggregated_mu_numerator, dim=0) / aggregated_T
        return vs_mus, vs_vars, aggregated_mu, aggregated_var

    def generation_x(self, z):
        xr_list = [self.decoders[f'view_{v}'](z) for v in range(self.num_views)]
        return xr_list

    def sample(self, mu, var):
        std = ops.sqrt(var)
        eps = self.normal_dist.sample(std.shape)
        return ops.mul(eps, std) + mu

    def construct(self, imv_data, mask=None):
        vs_mus, vs_vars, aggregated_mu, aggregated_var = self.inference_z(imv_data, mask)
        z_sample = self.sample(aggregated_mu, aggregated_var)
        xr_list = self.generation_x(z_sample)
        vade_z_sample = self.sample(aggregated_mu, aggregated_var)
        return z_sample, vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample

    @jit
    def vs_encode(self, sv_data, view_idx):
        latent_representation, _ = self.encoders[f'view_{view_idx}'](sv_data)
        xv_rec = self.decoders[f'view_{view_idx}'](latent_representation)
        return latent_representation, xv_rec


