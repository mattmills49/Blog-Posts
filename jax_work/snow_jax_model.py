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
from jax.scipy.special import expit

def load_data(X = 'X_array.pkl', Y = 'Y_array.pkl'):
    with open(X, 'rb') as f:
        X = pickle.load(f)
    
    with open(Y, 'rb') as f:
        Y = pickle.load(f)
    
    return {'X': jnp.asarray(X), 'Y': jnp.asarray(Y)}

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

class hurdleMLP(nn.Module):
    '''Predict both P(any snow) & E[snow]'''
    num_feats: int
    num_output: int
    batch_size: int
    
    def setup(self):
        self.W1 = self.param('W1', nn.initializers.normal(.1), (self.num_feats, self.num_feats))
        self.b1 = self.param('b1', nn.initializers.uniform(.1), (self.batch_size, self.num_feats))
        
        self.W2 = self.param('W2', nn.initializers.normal(.1), (self.num_feats, self.num_output))
        self.b2 = self.param('b2', nn.initializers.uniform(.1), (self.batch_size, self.num_output))
        ## Logistic Weights
        self.LW1 = self.param('LW1', nn.initializers.normal(.1), (self.num_feats, self.num_feats))
        self.Lb1 = self.param('Lb1', nn.initializers.uniform(.1), (self.batch_size, self.num_feats))
        
        self.LW2 = self.param('LW2', nn.initializers.normal(.1), (self.num_feats, self.num_output))
        self.Lb2 = self.param('Lb2', nn.initializers.uniform(.1), (self.batch_size, self.num_output))
    
    def __call__(self, inputs):
        x = inputs
        ## E[snow] Model
        ex = jnp.matmul(x, self.W1) + self.b1
        ex = jnp.matmul(ex, self.W2) + self.b2
        ## P(snow) Model
        px = jnp.matmul(x, self.LW1) + self.Lb1
        px = jnp.matmul(px, self.LW2) + self.Lb2
        px = expit(px)
        return ex, px
    
def hurdle_loss(params, x, y, model):
    
    def zero_log_loss(x, y):
        preds_ex, preds_px = model.apply(params, x)
        preds_ex = preds_ex[0, :]
        preds_px = preds_px[0, :]
        snow_ind = jnp.where(y > 0, 1, 0)
        ## $l(y, p) = p^{y} * (1 - p)^{1 - y}$
        ## $ll(y, p) = y * log(p) + (1 - y) * log(1 - p)$
        ## maximize the likelihood, so minimize the negative likelihood
        log_logistic = -jnp.sum(snow_ind * jnp.log(preds_px) + (1 - snow_ind) * jnp.log(1 - preds_px))
        ## We can't do boolean indexing in jax, but if we just multiply the resid by 0
        ## if there is no snow then those observations won't contribute to the likelihood
        #cont_resid = jnp.where(y > 0, y - preds_ex, 0)
        #log_normal = jnp.inner(cont_resid, cont_resid) / 2.0
        return log_logistic #+ log_normal
    
    return jnp.mean(jax.vmap(zero_log_loss)(x, y), axis = 0)

def get_batch(x, y, n = 20):
    batch_rows = np.random.randint(low = 0, high = x.shape[0], size = n)
    return x[batch_rows], y[batch_rows]

class log1p_transform():
    
    def __init__(self):
        self.name = 'log1p'
        
    def transform(self, x, y):
        x_ = x.copy()
        x_ = jnp.log1p(x_)
        
        y_ = y.copy()
        y_ = jnp.log1p(y_)
        
        return x_, y_
    
    def inv_transform(self, x, y):
        x_ = x.copy()
        x_ = jnp.exp(x_) - 1
        
        y_ = y.copy()
        y_ = jnp.exp(y_) - 1
        
def run_sgd(model, params, x, y, loss_fn, num_iter = 101):
    model_loss = ft.partial(loss_fn, model = model)
    
    tx = optax.adam(learning_rate = 0.05)
    opt_state = tx.init(params)
    loss_grad_fn = jax.value_and_grad(model_loss)

    for i in range(num_iter):
        x_batch, y_batch = get_batch(x = x, y = y)
        loss_val, grads = loss_grad_fn(params, x_batch, y_batch)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        grad_max = jax.tree_util.tree_map(lambda x: jnp.max(x), grads)
        if i % 10 == 0:
            print(f'Loss step {i}: {loss_val}')
            print(f'Max Gradient: {grad_max}')
    
    return params, loss_val
    

if __name__ == '__main__':
    
    data = load_data()
    x = data['X'][:, :-3]
    y = data['Y']
    
    transform = log1p_transform()
    x_log, y_log = transform.transform(x, y)
    
    key1, key2 = random.split(random.key(0))
    model = simpleMLP(num_feats = x.shape[1], 
                      num_output = y.shape[1], 
                      batch_size = 20)
    
    params = model.init(key1, x[:model.batch_size])
    print('Baseline Model')
    #baseline_params, baseline_loss = run_sgd(model, params, x_log, y_log, squared_loss)
    
    params2 = model.init(key1, x[:model.batch_size])
    print('Smooth Model')
    #smooth_params, smooth_loss = run_sgd(model, params2, x_log, y_log, smooth_squared_loss)
    
    model_hurdle = hurdleMLP(num_feats = x.shape[1], 
                             num_output = y.shape[1],
                             batch_size = 20)
    params_hurdle = model_hurdle.init(key1, x[:model.batch_size])
    print('Hurdle Model')
    hurdle_params, hurdle_loss_val = run_sgd(model_hurdle, params_hurdle, x, y, hurdle_loss)
    