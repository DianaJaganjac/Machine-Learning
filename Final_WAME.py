# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:52:52 2020

@author: Diana Jaganjac
"""

from keras import backend as K
from keras.optimizers import Optimizer 
import math


# line 1 + 2 of WAME initalising variables 

class WAME(Optimizer):
    def __init__(self, alpha = 0.9, scale_up = 1.2, scale_down = 0.1, zeta_min = 0.01, 
                 zeta_max = 100, lr = 0.001,
                 **kwargs):
        
        self.initial_t = kwargs.pop('t', 0.0)
        
        super(WAME, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name='alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.zeta_min = K.variable(zeta_min, name='zeta_min')
        self.zeta_max = K.variable(zeta_max, name='zeta_max')
        self.lr = K.variable(lr, name='lr')
        self.t = K.variable(self.initial_t, name='t')
        
        
    def get_updates(self, params, constraints, loss):
        
        grads = self.get_gradients(loss, params)
        
        shapes = [K.get_variable_shape(p) for p in params]
        
        old_grads = [K.zeros(shape) for shape in shapes]
        
        self.updates = []
        
        new_zeta = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='new_zeta_' + str(i))
              for (i, p) in enumerate(params)]
        
        Z = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='Z_' + str(i))
              for (i, p) in enumerate(params)]
        
        theta = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='theta_' + str(i))
              for (i, p) in enumerate(params)]


        for param, grad, old_grad, new_zeta, Z, theta in zip(params, grads, old_grads, new_zeta, Z, theta):
            
            new_step = K.switch(
                K.greater(grad * old_grad, 0),
                
                K.minimum(new_zeta * self.scale_up, self.zeta_max),
                
                K.maximum(new_zeta * self.scale_down, self.zeta_min)
            )
            
            Z_updated = ((self.alpha * Z) + ((1. - self.alpha) * new_step)) #was new_zeta before
            
            theta_updated = ((self.alpha * theta) + ((1. - self.alpha) * ((grad * self.t)**2)))
            
            new_t = (-self.lr * Z_updated * grad * (math.sqrt(1/theta_updated)))
            
            new_param = param + new_t 
            
            # Apply constraints
            if param in constraints:
                c = constraints[param]
                new_param = c(new_param)
                
                
            self.updates.append(K.update(param, new_param))
            self.updates.append(K.update(old_grad, grad))
            self.updates.append(K.update(new_zeta, new_step))
            self.updates.append(K.update(Z, Z_updated))
            self.updates.append(K.update(theta, theta_updated))
            
            

        return self.updates
    
    def get_config(self):
        config = {
            'alpha': float(K.get_value(self.alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'zeta_min': float(K.get_value(self.zeta_min)),
            'zeta_max': float(K.get_value(self.zeta_max)),
            'lr': float(K.get_value(self.lr)),
            't': float(K.get_value(self.t)),
        }
        
        base_config = super(WAME, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
    