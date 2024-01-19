import numpy as np
import pandas as pd
from plotnine import *
import itertools as it
import functools as ft
import pickle

import jax
from typing import Any, Callable, Sequence
from jax import random, numpy as jnp
import flax
from flax import linen as nn
import optax

def load_data(X = 'X_array.pkl', Y = 'Y_array.pkl'):
    with open(X, 'rb') as f:
        X = pickle.load(f)
    
    with open(Y, 'rb') as f:
        Y = pickle.load(f)
    
    return {'X': X, 'Y': Y}

class simpleMLP(nn.Module):
    num_feats: int
    num_output: int
    batch_size: int
    
    def setup(self):
        ## self.param: Declares and returns a parameter in this Module.
        ## nn.init..: Builds an initializer that returns real normally-distributed random arrays.
        self.W1 = self.param('W1', nn.initializers.normal(.1), (self.num_feats, self.num_feats))
        self.b1 = self.param('b1', nn.initializers.uniform(.1), (self.batch_size, self.num_feats))
        
        self.W2 = self.param('W2', nn.initializers.normal(.1), (self.num_feats, self.num_output))
        self.b2 = self.param('b2', nn.initializers.uniform(.1), (self.batch_size, self.num_output))
    
    def __call__(self, inputs):
        x = inputs
        ## first layer
        x = jnp.matmul(x, self.W1) + self.b1
        ## second layer
        x = jnp.matmul(x, self.W2) + self.b2
        return x
        
        

def squared_loss(params, x, y, model):
    '''Calculate the squared error loss for a matrix of snow predictions
    
    :param params: The model parameters
    :param x: the input matrix
    :param y: the dependent matrix
    :param model: the model to use to get predictions
    '''
    ## Define the squared loss for a single pair
    def squared_error(x, y):
        preds = model.apply(params, x)[0, :]
        results = jnp.inner(y - preds, y - preds) / 2.0
        return results
    ## vectorize the previous to compute the average of the loss on all samples
    return jnp.mean(jax.vmap(squared_error)(x, y), axis = 0)

def smooth_squared_loss(params, x, y, model, lam = 0.05):
    '''Calculate the squared error loss for a matrix of snow predictions
    
    :param params: The model parameters
    :param x: the input matrix
    :param y: the dependent matrix
    :param model: the model to use to get predictions
    '''
    ## Define the squared loss for a single pair
    def squared_error(x, y):
        preds = model.apply(params, x)[0, :]
        pred_diffs = jnp.diff(preds)
        loss = jnp.inner(y - preds, y - preds) / 2.0
        pen = lam * jnp.inner(pred_diffs, pred_diffs) / 2.0
        return loss + pen
    ## vectorize the previous to compute the average of the loss on all samples
    return jnp.mean(jax.vmap(squared_error)(x, y), axis = 0)

def get_batch(x, y, n = 20):
    batch_rows = np.random.randint(low = 0, high = x.shape[0], size = n)
    return x[batch_rows], y[batch_rows]

class log1p_transform():
    
    def __init__(self):
        self.name = 'log1p'
        
    def transform(self, x, y):
        x_ = x.copy()
        x_ = np.log1p(x_)
        
        y_ = y.copy()
        y_ = np.log1p(y_)
        
        return x_, y_
    
    def inv_transform(self, x, y):
        x_ = x.copy()
        x_ = np.exp(x_) - 1
        
        y_ = y.copy()
        y_ = np.exp(y_) - 1
        
def run_sgd(model, params, x, y, loss_fn):
    model_loss = ft.partial(loss_fn, model = model)
    
    tx = optax.adam(learning_rate = 0.05)
    opt_state = tx.init(params)
    loss_grad_fn = jax.value_and_grad(model_loss)

    for i in range(101):
        x_batch, y_batch = get_batch(x = x, y = y)
        loss_val, grads = loss_grad_fn(params, x_batch, y_batch)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i % 10 == 0:
            print(f'Loss step {i}: {loss_val}')
    
    return params, loss_val
    

if __name__ == '__main__':
    
    data = load_data()
    x = data['X'][:, :-3]
    y = data['Y']
    
    x_raw = x
    y_raw = y
    transform = log1p_transform()
    x, y = transform.transform(x, y)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    
    key1, key2 = random.split(random.key(0))
    model = simpleMLP(num_feats = x.shape[1], num_output = y.shape[1], batch_size = 20)
    params = model.init(key1, x[:model.batch_size])
    print('Baseline Model')
    baseline_params, baseline_loss = run_sgd(model, params, x, y, squared_loss)
    params2 = model.init(key1, x[:model.batch_size])
    print('Smooth Model')
    smooth_params, smooth_loss = run_sgd(model, params2, x, y, smooth_squared_loss)
    
    