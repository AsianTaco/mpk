import flax.traverse_util
import pytest
import jax
from jax import random as jr, numpy as jnp
from mpk import mpk_cnn_factory
from mpk.flax import MPK_layer
from flax import linen as nn
from typing import Any, Callable, Sequence


@pytest.fixture
def multipole_cnn():
    model_factory = mpk_cnn_factory.MultipoleCNNFactory(kernel_shape=[3, 3, 3],
                                                        polynomial_degrees=[0],
                                                        num_input_filters=1,
                                                        output_filters=None)
    return model_factory.build_flax_cnn_model(backend='scipy')


class BenchmarkCNN(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=1, kernel_size=(3, 3, 3), padding='CIRCULAR')(x)
        return x


# define some other module that uses the layer
class MPK_EmbedNet(nn.Module):
    """A demonstration of using a MultiPole Kernel (MPK) embedding"""
    filters: Sequence[int]
    multipole_tomo1: MPK_layer
    act: Callable = nn.swish
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        # embed the data in multipoles
        return self.multipole_tomo1(x)


# TODO: Get rid of this code duplication of MultipoleConv.get_kernel(self)
def get_kernel(num_params, kernel_weights, multipole_kernels):
    if num_params == 1:
        return kernel_weights * multipole_kernels

    return jnp.dot(jnp.transpose(multipole_kernels), kernel_weights).squeeze()


def test_convolution(multipole_cnn):
    # Arrange
    test_input = jnp.arange(125).reshape([5, 5, 5])
    key = jr.PRNGKey(42)

    multipole_cnn_params = multipole_cnn.init(key, test_input)
    kernel_weights = flax.traverse_util.flatten_dict(multipole_cnn_params)[('params', 'kernel_weights')]
    model_kernel = get_kernel(multipole_cnn.num_params, kernel_weights, multipole_cnn.multipole_kernels)
    kernel_bias = flax.traverse_util.flatten_dict(multipole_cnn_params)[('params', 'bias')]

    benchmark_model = BenchmarkCNN()
    dummy_params = benchmark_model.init(key, test_input.reshape([1, 5, 5, 5, 1]))
    tmp_param = flax.traverse_util.flatten_dict(dummy_params)
    new_kernel_shape = tmp_param[('params', 'Conv_0', 'kernel')].shape
    tmp_param[('params', 'Conv_0', 'kernel')] = model_kernel.reshape(new_kernel_shape)
    tmp_param[('params', 'Conv_0', 'bias')] = kernel_bias
    benchmark_cnn_params = flax.traverse_util.unflatten_dict(tmp_param)

    # Act
    multipole_cnn_out = multipole_cnn.apply(multipole_cnn_params, test_input).squeeze()
    output_benchmark = benchmark_model.apply(benchmark_cnn_params, test_input.reshape([1, 5, 5, 5, 1]))
    cleaned_benchmark_output = output_benchmark.squeeze()

    # Assert
    assert multipole_cnn_out.shape == cleaned_benchmark_output.shape
    assert jnp.allclose(cleaned_benchmark_output, multipole_cnn_out)


def test_training(multipole_cnn):
    n_samples = 20
    test_input = jnp.arange(125).reshape([5, 5, 5])
    true_key = jr.PRNGKey(0)
    init_key = jr.PRNGKey(42)

    true_params = multipole_cnn.init(true_key, test_input)
    params = multipole_cnn.init(init_key, test_input)

    # Generate samples TODO: Add noise maybe?
    key_sample, key_noise = jr.split(true_key)
    x_samples = jr.normal(key_sample, (n_samples, 5, 5, 5))

    y_samples = jnp.zeros((n_samples, 5, 5, 5))

    for i, x in enumerate(x_samples):
        y_samples = y_samples.at[i, :, :, :].set(multipole_cnn.apply(true_params, x).squeeze())

    @jax.jit
    def mse(params, x_batched, y_batched):
        # Define the squared loss for a single pair (x,y)
        def squared_error(x, y):
            pred = multipole_cnn.apply(params, x)
            return jnp.sum(jnp.inner(y - pred, y - pred) / 2.0)

        # Vectorize the previous to compute the average of the loss on all samples.
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

    learning_rate = 1e-4
    print('Loss for "true" W,b: ', mse(true_params, x_samples, y_samples))
    loss_grad_fn = jax.value_and_grad(mse)

    @jax.jit
    def update_params(model_params, learning_rate, gradient):
        model_params = jax.tree_util.tree_map(
            lambda p, g: p - learning_rate * g, model_params, gradient)
        return model_params

    initial_loss = mse(params, x_samples, y_samples)

    for i in range(10000):
        loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
        params = update_params(params, learning_rate, grads)

    final_loss = mse(params, x_samples, y_samples)

    assert final_loss < initial_loss

    print(f"final loss: {final_loss}")
    print(f"final parameters: {params}")
    print(f"true parameters: {true_params}")


@pytest.mark.skip(reason="Not implemented yet")
def test_higher_order():
    model_factory = mpk_cnn_factory.MultipoleCNNFactory(kernel_shape=[3, 3, 3],
                                                        polynomial_degrees=[0, 1],
                                                        num_input_filters=1,
                                                        output_filters=None)
    multipole_conv_model = model_factory.build_flax_cnn_model()

    test_input = jnp.arange(125).reshape([5, 5, 5])
    key = jr.PRNGKey(42)

    # TODO: add more elaborate test
    multipole_cnn_params = multipole_conv_model.init(key, test_input)


def test_embedding():
    # example: here we want to apply an MPK residual layer to an input of shape (64, 64, 4)
    dtype = jnp.float32
    kernel_size = 7
    polynomial_degrees = [0, 1, 2]
    mpk_strides = [1, 1]
    mpk_input_filters = [4, 6]  # input has 4 filters, output from \ell=[0,1,2] has 6
    # TODO: can we get the code to detect the output filters automatically / have a lookup for the output size for given \ell ""

    # define layer
    mpk_layer = mpk_cnn_factory.MultipoleCNNFactory(
        kernel_shape=(kernel_size, kernel_size),
        polynomial_degrees=polynomial_degrees,
        output_filters=None,
        dtype=dtype)

    # initialise the model
    model_key = jr.PRNGKey(44)
    model = MPK_EmbedNet(
        filters=3,
        # rest of network
        multipole_tomo1=MPK_layer(
            multipole_layers=[mpk_layer.build_flax_cnn_model(num_input_filters=f,
                                                             strides=mpk_strides,
                                                             pad_size=None) for i, f in enumerate(mpk_input_filters)],
            act=nn.swish),
    )

    # apply it to some dummy inputs
    w = model.init(model_key, jnp.ones((64, 64, 4)))
    print("shape of outputs from model", model.apply(w, jnp.ones((64, 64, 4))).shape)
