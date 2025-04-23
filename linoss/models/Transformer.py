import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


def causal_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


def sinusoidal_encoding(seq_len, d_model):
    pos = jnp.arange(seq_len)[:, None]
    i = jnp.arange(d_model)[None, :]

    angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates

    pe = jnp.where(i % 2 == 0, jnp.sin(angle_rads), jnp.cos(angle_rads))

    return pe


class Embedding(eqx.Module):
    linear_layer: eqx.nn.Linear

    def __init__(self, input_dim, model_dim, *, key):
        self.linear_layer = eqx.nn.Linear(input_dim, model_dim, key=key)

    def __call__(self, x):
        x = jax.vmap(self.linear_layer)(x)
        pos_encoding = sinusoidal_encoding(x.shape[0], x.shape[1])

        return x + pos_encoding


class EncoderBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    norm2: eqx.nn.LayerNorm

    def __init__(self, dim, num_heads, mlp_dim, *, key):
        k1, k2 = jax.random.split(key, 2)
        self.attn = eqx.nn.MultiheadAttention(num_heads, dim, key=k1)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.mlp = eqx.nn.MLP(dim, dim, mlp_dim, 1, key=k2)
        self.norm2 = eqx.nn.LayerNorm(dim)

    def __call__(self, x):
        x = jax.vmap(self.norm1)(x + self.attn(x, x, x))
        x = jax.vmap(self.norm2)(x + jax.vmap(self.mlp)(x))

        return x


class DecoderBlock(eqx.Module):
    self_attn: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    cross_attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    norm3: eqx.nn.LayerNorm

    def __init__(self, dim, num_heads, mlp_dim, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.self_attn = eqx.nn.MultiheadAttention(num_heads, dim, key=k1)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.cross_attn = eqx.nn.MultiheadAttention(num_heads, dim, key=k2)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.mlp = eqx.nn.MLP(dim, dim, mlp_dim, 1, key=k3)
        self.norm3 = eqx.nn.LayerNorm(dim)

    def __call__(self, x, enc_out):
        mask = causal_mask(x.shape[0])
        x = jax.vmap(self.norm1)(x + self.self_attn(x, x, x, mask=mask))
        x = jax.vmap(self.norm2)(x + self.cross_attn(x, enc_out, enc_out))
        x = jax.vmap(self.norm3)(x + jax.vmap(self.mlp)(x))

        return x


class EncoderStack(eqx.Module):
    """
    Simplified encoder-only transformer architecture
    """

    start_token: jax.Array = eqx.field(init=False)
    encoder_embedding: Embedding
    encoder_blocks: list
    linear_layer: eqx.nn.Linear
    classification: bool = False
    linear_output: bool
    stateful: bool = False
    nondeterministic: bool = False
    lip2: bool = False

    def __init__(
        self,
        encoder_depth,
        input_dim,
        model_dim,
        output_dim,
        num_heads,
        classification,
        linear_output,
        *,
        key,
    ):
        self.classification = classification
        if self.classification:
            raise NotImplementedError
        self.linear_output = linear_output

        ks, kme, ke, kl = jr.split(key, 4)
        self.start_token = jax.random.normal(ks, (output_dim,))
        self.encoder_embedding = Embedding(input_dim, model_dim, key=kme)
        encoder_keys = jr.split(ke, encoder_depth)
        mlp_dim = 4 * model_dim
        self.encoder_blocks = [
            EncoderBlock(model_dim, num_heads, mlp_dim, key=k) for k in encoder_keys
        ]
        self.linear_layer = eqx.nn.Linear(model_dim, output_dim, key=kl)

    def __call__(self, enc_in, dec_in):
        x = self.encoder_embedding(enc_in)
        for encoder in self.encoder_blocks:
            x = encoder(x)

        output = jax.vmap(self.linear_layer)(x)

        if not self.linear_output:
            output = jax.nn.tanh(output)

        return output


class Transformer(eqx.Module):
    """
    Vanilla encoder-decoder transformer architecture
    """

    start_token: jax.Array = eqx.field(init=False)
    encoder_embedding: Embedding
    decoder_embedding: Embedding
    encoder_blocks: list
    decoder_blocks: list
    linear_layer: eqx.nn.Linear
    classification: bool = False
    linear_output: bool
    stateful: bool = False
    nondeterministic: bool = False
    lip2: bool = False

    def __init__(
        self,
        encoder_depth,
        decoder_depth,
        input_dim,
        model_dim,
        output_dim,
        num_heads,
        classification,
        linear_output,
        *,
        key,
    ):
        self.classification = classification
        if self.classification:
            raise NotImplementedError(
                "Current Transformer implementation designed for regressive outputs"
            )
        self.linear_output = linear_output

        ks, kme, kmd, ke, kd, kl = jr.split(key, 6)
        self.start_token = jax.random.normal(ks, (output_dim,))
        self.encoder_embedding = Embedding(input_dim, model_dim, key=kme)
        self.decoder_embedding = Embedding(output_dim, model_dim, key=kmd)
        encoder_keys = jr.split(ke, encoder_depth)
        decoder_keys = jr.split(kd, decoder_depth)
        mlp_dim = 4 * model_dim
        self.encoder_blocks = [
            EncoderBlock(model_dim, num_heads, mlp_dim, key=k) for k in encoder_keys
        ]
        self.decoder_blocks = [
            DecoderBlock(model_dim, num_heads, mlp_dim, key=k) for k in decoder_keys
        ]
        self.linear_layer = eqx.nn.Linear(model_dim, output_dim, key=kl)

    def encode(self, enc_in):
        x = self.encoder_embedding(enc_in)
        for encoder in self.encoder_blocks:
            x = encoder(x)

        return x

    def decode(self, dec_in, enc_out):
        x = self.decoder_embedding(dec_in)
        for decoder in self.decoder_blocks:
            x = decoder(x, enc_out)

        return x

    def __call__(self, enc_in, dec_in):
        enc_out = self.encode(enc_in)
        dec_in = jnp.concatenate([self.start_token[None, :], dec_in[:-1]], axis=0)
        output = self.decode(dec_in, enc_out)
        output = jax.vmap(self.linear_layer)(output)
        if not self.linear_output:
            output = jax.nn.tanh(output)
        return output

    def autoregressive_inference(self, enc_in):
        seq_len = enc_in.shape[0]
        enc_out = self.encode(enc_in)
        dec_seq = self.start_token[None, :]

        for i in range(seq_len):
            x = self.decode(dec_seq, enc_out)
            next_vec = self.linear_layer(x[-1])

            if not self.linear_output:
                next_vec = jax.nn.tanh(next_vec)

            dec_seq = jnp.concatenate([dec_seq, next_vec[None, :]], axis=0)

        return dec_seq[1:]
