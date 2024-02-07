import numpy as np
import torch
from scipy import linalg

def calculate_FID(real, fake, netD):
    real_data = real.float().to('cuda')
    generated_data = fake.float().to('cuda')

    with torch.no_grad():
        act_real = netD(real_data, interm=True)
        act_generated = netD(generated_data, interm=True)

    # Reshaping the activations and treating each time-series data point as an individual instance
    act_real = act_real.reshape(-1, act_real.shape[-1])
    act_generated = act_generated.reshape(-1, act_generated.shape[-1])

    act_real = act_real.cpu().numpy()
    act_generated = act_generated.cpu().numpy()

    # Calculate mean and covariance statistics
    mu_real, sigma_real = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    mu_gen, sigma_gen = act_generated.mean(axis=0), np.cov(act_generated, rowvar=False)

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError(
            "(mu1, sigma1) should have exactly the same shape as (mu2, sigma2)."
        )

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            "WARNING: fid calculation produces singular product; adding {} to diagonal of cov estimates"
            .format(eps))

        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean