import torch


def compute_kl_div(dist_q, dist_p, n_samples=100):
    samples_q = dist_q.sample((n_samples,))
    kl = torch.mean(dist_q.log_prob(samples_q) - dist_p.log_prob(samples_q))
    return kl


def compute_ll(dist_lik, data_samples):
    total_ll = torch.sum(dist_lik.log_prob(data_samples), dim=-1)
    return torch.mean(total_ll)


def elbo_loss(qs, ps, ys, likelihood):
    total_kl = torch.tensor(0.0)
    for q, p in zip(qs, ps):
        total_kl += compute_kl_div(q, p)
    ll = compute_ll(likelihood, ys)
    neg_elbo = total_kl - ll
    return neg_elbo
