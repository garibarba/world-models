"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

def gmm_loss(batch, mus, logsigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
    """ Computes the gmm loss.

    Compute an upper bound to the KL divergence of the normal observation and
    the gmm prediction KL(N||GMM) <= sum_i(pi_i * KL(N||N_i)) of batch under
    the GMM model described by mus, sigmas, pi. Precisely, with bs1, bs2, ...
    the sizes of the batch dimensions (several batch dimension are useful
    when you have both a batch axis and a time step axis), gs the number of
    mixtures and fs the number of features.

    :args batch: ((bs1, bs2, *, fs),) * 2 tuple of torch tensor for (mus, logsigmas)
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args logsigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    batch_mus, batch_sigmas = batch[0].unsqueeze(-2), batch[1].unsqueeze(-2).exp()
    sigmas = logsigmas.exp()
    pis = torch.softmax(logpi, dim=-1)

    kl_divs_normal = 0.5 * ((batch_sigmas / sigmas).sum(-1)
                            + ((mus - batch_mus)**2 / sigmas).sum(-1)
                            - sigmas.shape[-1]
                            + (sigmas.prod(-1) / batch_sigmas.prod(-1)).log())
    kl_divs_gmm_upper_bound = (kl_divs_normal * pis).sum(-1)

    if reduce:
        return torch.mean(kl_divs_gmm_upper_bound)
    return kl_divs_gmm_upper_bound

class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, logsig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logsigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        logsigmas = gmm_outs[:, :, stride:2 * stride]
        logsigmas = logsigmas.view(seq_len, bs, self.gaussians, self.latents)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, logsigmas, logpi, rs, ds

class MDRNNCell(_MDRNNBase):
    """ MDRNN model for one step forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
        """ ONE STEP forward.

        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        logsigmas = out_full[:, stride:2 * stride]
        logsigmas = logsigmas.view(-1, self.gaussians, self.latents)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpis = f.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, logsigmas, logpis, r, d, next_hidden

class FMDRNNCell(MDRNNCell):
    """ Filtering MDRNNCell. Keeps previous output for filtering. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_gmm = None
        self._gmm_sampler = MyGMMSampler.apply

    def reset(self):
        self.prev_gmm = None

    def forward(self, action, latents, hidden):
        """ ONE STEP forward.

        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: 2 * ((BSIZE, LSIZE),) tuple of torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor
        """ 
        if self.prev_gmm:
            mus, logsigmas, logpi = self._filter(latents)
            latent_input = self._gmm_sampler(mus, logsigmas, logpi)
        else:
            latent_input = latents[0] # deterministic. TODO: sampled?
        latent_input.detach_()
        mus, logsigmas, logpi, r, d, next_hiden = super().forward(action, latent_input, hidden)
        self.prev_gmm = (mus, logsigmas, logpi)

        return mus, logsigmas, logpi, r, d, next_hiden

    def _filter(self, latents):
        lat_mus, lat_logsigmas = tuple(map(lambda x: torch.unsqueeze(x, dim=-2),
                                           latents))
        gmm_mus, gmm_logsigmas, gmm_logpi = self.prev_gmm

        mus, logsigmas, = prod_multivariate_normals((lat_mus, lat_logsigmas),
                                                    (gmm_mus, gmm_logsigmas))
        joint_logconstant = log_constant_term_multivariate_normal(mus, logsigmas)
        lat_logconstant = log_constant_term_multivariate_normal(lat_mus, lat_logsigmas)
        gmm_logconstant = log_constant_term_multivariate_normal(gmm_mus, gmm_logsigmas)

        logpi = gmm_logpi + lat_logconstant + gmm_logconstant - joint_logconstant

        return mus, logsigmas, logpi

def prod_multivariate_normals(normal_1, normal_2):
    """
    mean and logstd from the normal resulting of multiplying 2 normals
    """
    mus_1, logsigmas_1 = normal_1
    mus_2, logsigmas_2 = normal_2

    sigmas_1_sq, sigmas_2_sq = logsigmas_1.exp() ** 2, logsigmas_2.exp() ** 2

    mus = (mus_1 * sigmas_2_sq + mus_2 * sigmas_1_sq) / (sigmas_1_sq + sigmas_2_sq)
    logsigmas = - 0.5 * torch.log(1 / sigmas_1_sq + 1 / sigmas_2_sq)

    return mus, logsigmas

def log_constant_term_multivariate_normal(mus, logsigmas):
    """
    The constant term in a multivariate normal distribution refers to
    the term including all terms that do not depend on x.
    """
    sigmas = logsigmas.exp()
    d = mus.shape[-1]

    return - 0.5 * (d * np.log(2 * np.pi)
                    + (sigmas ** 2).prod(dim=-1)
                    - ((mus / sigmas) ** 2).sum(-1)
                    )

def log_prob_multivariate_normal(x, mus, sigma_logits): 
  sigmas = sigma_logits.exp()
  return - 0.5 * (x.shape[-1] * np.log(2 * np.pi)
                  + (sigmas ** 2).prod(-1).log()
                  + (((x - mus) / sigmas) ** 2).sum(-1))

class MyGMMSampler(torch.autograd.Function):
  @staticmethod
  def forward(ctx, mus, sigma_logits, pi_logits):
    k = torch.distributions.Categorical(logits=pi_logits).sample()
    dims = sigma_logits.shape
    meshindexes = torch.meshgrid(
        *[torch.arange(d) for d in dims[:-2]]) if len(dims) > 2 else tuple()
    indexes = meshindexes + (k,) + (slice(None),)
    x = torch.randn_like(mus[indexes]) * torch.exp(sigma_logits[indexes]) + mus[indexes] # sample
    ctx.save_for_backward(mus, sigma_logits, pi_logits, x)
    return x

  @staticmethod
  def backward(ctx, grad_output):
    mus, sigma_logits, pi_logits, x = ctx.saved_tensors

    log_probs = torch.empty_like(pi_logits)
    for i in range(log_probs.shape[-1]):
        log_probs[..., i] = log_prob_multivariate_normal(x, mus[..., i, :], sigma_logits[..., i, :])
    class_posterior_logits = log_probs + pi_logits
    class_posterior_probs = torch.softmax(class_posterior_logits, -1)
    grad_samples = grad_output.unsqueeze(-2) * \
        class_posterior_probs.unsqueeze(-1)
    assert grad_samples.shape == mus.shape
    assert grad_samples.shape == sigma_logits.shape

    with torch.enable_grad():
      sigmas = torch.exp(sigma_logits)
      normal_samples = (x.unsqueeze(-2) - mus) / sigmas
      normal_samples.detach_()
      x_resamples = normal_samples * sigmas + mus
      grad_mus, grad_sigma_logits = torch.autograd.grad(
          x_resamples, [mus, sigma_logits], grad_samples)

    grad_pi_logits = (grad_output * x).sum(-1, keepdim=True) * (
        class_posterior_probs - torch.softmax(pi_logits, -1))

    return grad_mus, grad_sigma_logits, grad_pi_logits

class FMDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.cell = FMDRNNCell(latents, actions, hiddens, gaussians)

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, logsig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logsigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)
        self.cell.reset()
        hidden = (actions.new(bs, self.hiddens).zero_(),) * 2
        outputs = []
        for action, mu, logsigma in zip(actions, *latents):
            cell_returns = self.cell(action, (mu, logsigma), hidden)
            cell_outputs, hidden = cell_returns[:-1], cell_returns[-1]
            outputs.append(cell_outputs)
        
        mus, logsigmas, logpi, rs, ds = tuple(map(torch.stack,
                                                  zip(*outputs)
                                                  ))
        
        return mus, logsigmas, logpi, rs, ds
