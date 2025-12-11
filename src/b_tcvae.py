import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from tensorflow_probability.substrates.jax import distributions as tfd


"""
There's a typo in most B-TCVAE implementations on github, so I thought I'd make a 
quick gist of a working B-TCVAE.

The problem is with the log importance weight matrix -- most implementations don't 
sample correct diagonals, so this implementation uses straightforward aranges to 
sample the correct diagonals.

See this pull request for original find by another user:
https://github.com/rtqichen/beta-tcvae/pull/1
"""


def _log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = jnp.full((batch_size, batch_size), 1 / M, dtype=jnp.float32)
    # this is what is fixed
    W = W.at[jnp.arange(batch_size), jnp.arange(batch_size)].set(1 / N)
    W = W.at[jnp.arange(batch_size - 1), jnp.arange(1, batch_size)].set(strat_weight)
    W = W.at[batch_size - 1, 0].set(strat_weight)
    #######################
    return jnp.log(W)


class BetaTCVAE(nn.Module):
    latent_size: int
    hidden_size: int
    num_hidden: int
    dataset_size: int
    beta: float = 6.0

    @nn.compact
    def __call__(self, x, train_step, is_training=True):
        batch_size = x.shape[0]

        # encoder
        x_in = x
        encoder_out = Encoder(self.latent_size, self.hidden_size, self.num_hidden)(x)
        z_mean, z_logvar = jnp.split(encoder_out, 2, axis=-1)

        # sampling
        q_dist = tfd.Normal(loc=z_mean, scale=jnp.exp(0.5 * z_logvar))
        if is_training:
            z_quantized = q_dist.sample(seed=self.make_rng("dist"))
        else:
            z_quantized = z_mean

        # decoder
        reconstruction = Decoder(x.shape[-1], self.hidden_size, self.num_hidden)(z_quantized)
        logpx = (
            tfd.Normal(loc=reconstruction, scale=jnp.ones_like(reconstruction))
            .log_prob(x_in.astype(jnp.float32))
            .sum(1)
        )

        # loss
        prior_dist = tfd.Normal(
            jnp.zeros_like(z_mean), jnp.ones_like(z_logvar)
        )
        logpz = prior_dist.log_prob(z_quantized).sum(1)
        logqz_condx = q_dist.log_prob(z_quantized).sum(1)

        _logqz = tfd.Normal(
            jnp.expand_dims(z_mean, 0),
            jnp.exp(0.5 * jnp.expand_dims(z_logvar, 0)),
        ).log_prob(jnp.expand_dims(z_quantized, 1))

        logiw_matrix = _log_importance_weight_matrix(
            batch_size, self.dataset_size
        )
        logqz = jax.scipy.special.logsumexp(
            logiw_matrix + _logqz.sum(2), axis=1, keepdims=False
        )
        logqz_prodmarginals = jax.scipy.special.logsumexp(
            logiw_matrix.reshape(batch_size, batch_size, 1) + _logqz,
            axis=1,
            keepdims=False,
        ).sum(1)

        anneal = 1.0 - optax.cosine_decay_schedule(1.0, 5000)(train_step)
        modified_elbo = (
                logpx
                - (logqz_condx - logqz)
                - anneal * self.beta * (logqz - logqz_prodmarginals)
                - anneal * (logqz_prodmarginals - logpz)
        )
        elbo_loss = -jnp.mean(modified_elbo)

        return reconstruction, elbo_loss, z_mean, z_logvar