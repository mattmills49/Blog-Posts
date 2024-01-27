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
    
    Parameters
    ----------
    params : dict
        The model parameters
    x : jax array
        the input matrix
    y : jax array
        the dependent matrix
    model : flax model
        the model to use to get predictions
        
    Returns
    -------
    The mean squared error across the observations
    '''
    ## Define the squared loss for a single pair
    def squared_error(x, y):
        preds = model.apply(params, x)[0, :]
        results = jnp.inner(y - preds, y - preds) / 2.0
        return results
    ## vectorize the previous to compute the average of the loss on all samples
    return jnp.mean(jax.vmap(squared_error)(x, y), axis = 0)


def smooth_squared_loss(params, x, y, model, lam = 0.05):
    '''Calculate the squared error loss plus a smoothness penalty for a matrix of snow predictions
    
    Parameters
    ----------
    params : dict
        The model parameters
    x : jax array
        the input matrix
    y : jax array
        the dependent matrix
    model : flax model
        the model to use to get predictions
    lam : float
        The weighting for the smoothness penalty portion of the loss function
        
    Returns
    -------
    The mean loss value across the observations
    '''
    ## Define the squared loss for a single pair
    def squared_error(x, y):
        preds = model.apply(params, x)[0, :]
        loss = jnp.inner(y - preds, y - preds) / 2.0
        ## The sum the squared difference between the successive predictions
        pred_diffs = jnp.diff(preds)
        pen = lam * jnp.inner(pred_diffs, pred_diffs) / 2.0
        return loss + pen
    ## vectorize the previous to compute the average of the loss on all samples
    return jnp.mean(jax.vmap(squared_error)(x, y), axis = 0)


class expMLP_(nn.Module):
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
        ## force positive
        x = jnp.exp(x)
        return x
    
class expMLP(nn.Module):
    num_feats: int
    num_output: int
    batch_size: int
    
    def setup(self):
        ## self.param: Declares and returns a parameter in this Module.
        ## nn.init..: Builds an initializer that returns real normally-distributed random arrays.
        self.W1 = self.param('W1', nn.initializers.normal(.1), (self.num_feats, self.num_output))
        self.b1 = self.param('b1', nn.initializers.uniform(.1), (1, 1))
        
    def __call__(self, inputs):
        x = inputs
        ## first layer
        x = jnp.matmul(x, self.W1) + self.b1
        ## force positive
        x = jnp.exp(x)
        return x
    
    
def exp_loss(params, x, y, model, lam = 0.05):
    '''Calculate the negative maximum likelihood for a matrix of snow predictions for an exponential distribution
    
    Parameters
    ----------
    params : dict
        The model parameters
    x : jax array
        the input matrix
    y : jax array
        the dependent matrix
    model : flax model
        the model to use to get predictions
        
    Returns
    -------
    The mean squared error across the observations
    '''
    ## Define the squared loss for a single pair
    def neg_exp_log_likelihood(x, y):
        preds = model.apply(params, x)[0, :]
        ## match scale parameter of scipy.stats.expon
        ## pdf = (1 / beta) * exp(-x / beta)
        ## beta = predicted value
        ## x = the observed value aka y
        ## ln(pdf) = ln(1 / beta) + ln(exp(-x / beta))
        ## ln(pdf) = -ln(beta) - x / beta
        ll = -jnp.log(preds) - y / preds
        pred_diffs = jnp.diff(preds)
        pen = lam * jnp.inner(pred_diffs, pred_diffs) / 2.0
        ## The optimization is minimizing our loss function, but we are 
        ## maximizing the log likelihood so we have to minimzize the negative
        ## log likehood
        nll = jnp.sum(-ll) + pen
        return nll
    ## vectorize the previous to compute the average of the loss on all samples
    return jnp.mean(jax.vmap(neg_exp_log_likelihood)(x, y), axis = 0)

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
        x_batch, y_batch = get_batch(x = x, y = y, n = 120)
        loss_val, grads = loss_grad_fn(params, x_batch, y_batch)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        grad_max = jax.tree_util.tree_map(lambda x: jnp.max(x), grads)
        max_pred = jnp.max(model.apply(params, x_batch))
        if i % 10 == 0:
            print(f'Loss step {i}: {loss_val}')
            print(f'Max Gradient: {grad_max}')
            print(f'Max Prediction: {max_pred}')
        
        if jnp.isnan(loss_val):
            break
    
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
    
    model_exp = expMLP(num_feats = x.shape[1], 
                       num_output = y.shape[1], 
                       batch_size = 20)
    params_exp = model_exp.init(key1, x[:model.batch_size])
    params_exp_fit, loss_val_exp = run_sgd(model_exp, params_exp, x, y, exp_loss, num_iter = 101)
    
    # model_hurdle = hurdleMLP(num_feats = x.shape[1], 
    #                          num_output = y.shape[1],
    #                          batch_size = 20)
    # params_hurdle = model_hurdle.init(key1, x[:model.batch_size])
    # print('Hurdle Model')
    # hurdle_params, hurdle_loss_val = run_sgd(model_hurdle, params_hurdle, x, y, hurdle_loss)
    
## the issue could be two fold:
# 1. Negative values in y
#+ 2. all 0 values
