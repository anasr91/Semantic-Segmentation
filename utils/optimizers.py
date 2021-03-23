#
# from termcolor import colored
# import tensorflow as tf
# import numpy as np
# import math
# from random import sample
# import torch
#
# from torch.optim import Optimizer
#
# K = tf.keras.backend
# Optimizer = tf.keras.optimizers.Optimizer
#
#
# def warn_str():
#     return colored('WARNING: ', 'red')
#
#
# class AdamW(Optimizer):
#
#     def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
#                  amsgrad=False, batch_size=32, total_iterations=0,
#                  weight_decays=None, lr_multipliers=None,
#                  use_cosine_annealing=False, eta_min=0, eta_max=1,
#                  t_cur=0, init_verbose=True, name='AdamW', **kwargs):
#         self.initial_decay = kwargs.pop('decay', 0.0)
#         self.epsilon = kwargs.pop('epsilon', K.epsilon())
#         learning_rate = kwargs.pop('lr', learning_rate)
#         eta_t = kwargs.pop('eta_t', 1.)
#         super(AdamW, self).__init__(name, **kwargs)
#
#         with K.name_scope(self.__class__.__name__):
#             self.iterations = K.variable(0, dtype='int64', name='iterations')
#             self.learning_rate = K.variable(learning_rate, name='learning_rate')
#             self.beta_1 = K.variable(beta_1, name='beta_1')
#             self.beta_2 = K.variable(beta_2, name='beta_2')
#             self.decay = K.variable(self.initial_decay, name='decay')
#             self.batch_size = K.variable(batch_size, dtype='int64',
#                                          name='batch_size')
#             self.total_iterations = K.variable(total_iterations, dtype='int64',
#                                                name='total_iterations')
#             self.eta_min = K.constant(eta_min, name='eta_min')
#             self.eta_max = K.constant(eta_max, name='eta_max')
#             self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
#             self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
#
#         self.amsgrad = amsgrad
#         self.lr_multipliers = lr_multipliers
#         self.weight_decays = weight_decays
#         self.init_verbose = init_verbose
#         self.use_cosine_annealing = use_cosine_annealing
#
#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = [K.update_add(self.iterations, 1)]
#         self.updates.append(K.update_add(self.t_cur, 1))
#
#         lr = self.learning_rate
#         if self.initial_decay > 0:
#             lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
#                                                       K.dtype(self.decay))))
#
#         t = K.cast(self.iterations, K.floatx()) + 1
#         lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
#                      (1. - K.pow(self.beta_1, t)))
#
#         ms = [K.zeros(K.int_shape(p),
#                       dtype=K.dtype(p),
#                       name='m_' + str(i))
#               for (i, p) in enumerate(params)]
#         vs = [K.zeros(K.int_shape(p),
#                       dtype=K.dtype(p),
#                       name='v_' + str(i))
#               for (i, p) in enumerate(params)]
#
#         if self.amsgrad:
#             vhats = [K.zeros(K.int_shape(p),
#                              dtype=K.dtype(p),
#                              name='vhat_' + str(i))
#                      for (i, p) in enumerate(params)]
#         else:
#             vhats = [K.zeros(1, name='vhat_' + str(i))
#                      for i in range(len(params))]
#         self.weights = [self.iterations] + ms + vs + vhats
#
#         total_iterations = K.get_value(self.total_iterations)
#         if total_iterations == 0:
#             print(warn_str() + "'total_iterations'==0, must be !=0 to use "
#                   + "cosine annealing and/or weight decays; "
#                   + "proceeding without either")
#         # Schedule multiplier
#         if self.use_cosine_annealing and total_iterations != 0:
#             t_frac = K.cast(self.t_cur / (self.total_iterations + 1), 'float32')
#             self.eta_t = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
#                          (1 + K.cos(np.pi * t_frac))
#             if self.init_verbose:
#                 print('Using cosine annealing learning rates')
#         self.lr_t = lr * self.eta_t  # for external tracking
#
#         for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
#             # Learning rate multipliers
#             multiplier_name = None
#             if self.lr_multipliers:
#                 multiplier_name = [mult_name for mult_name in self.lr_multipliers
#                                    if mult_name in p.name]
#             new_lr = lr_t
#             if multiplier_name:
#                 new_lr = new_lr * self.lr_multipliers[multiplier_name[0]]
#                 if self.init_verbose:
#                     print('{} learning rate set for {} -- {}'.format(
#                         '%.e' % K.get_value(new_lr), p.name.split('/')[0], new_lr))
#             elif not multiplier_name and self.init_verbose:
#                 print('No change in learning rate {} -- {}'.format(
#                     p.name, K.get_value(new_lr)))
#
#             m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
#             v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
#             if self.amsgrad:
#                 vhat_t = K.maximum(vhat, v_t)
#                 p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
#                 self.updates.append(K.update(vhat, vhat_t))
#             else:
#                 p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
#
#             # Weight decays
#             if p.name in self.weight_decays.keys() and total_iterations != 0:
#                 wd = self.weight_decays[p.name]
#                 wd_normalized = wd * K.cast(
#                     K.sqrt(self.batch_size / self.total_iterations), 'float32')
#                 p_t = p_t - self.eta_t * wd_normalized * p
#                 if self.init_verbose:
#                     print('{} weight decay set for {}'.format(
#                         K.get_value(wd_normalized), p.name))
#
#             self.updates.append(K.update(m, m_t))
#             self.updates.append(K.update(v, v_t))
#             new_p = p_t
#
#             # Apply constraints.
#             if getattr(p, 'constraint', None) is not None:
#                 new_p = p.constraint(new_p)
#
#             self.updates.append(K.update(p, new_p))
#         return self.updates
#
#     def get_config(self):
#         config = {'learning_rate': float(K.get_value(self.learning_rate)),
#                   'beta_1': float(K.get_value(self.beta_1)),
#                   'beta_2': float(K.get_value(self.beta_2)),
#                   'decay': float(K.get_value(self.decay)),
#                   'batch_size': int(K.get_value(self.batch_size)),
#                   'total_iterations': int(K.get_value(self.total_iterations)),
#                   'weight_decays': self.weight_decays,
#                   'lr_multipliers': self.lr_multipliers,
#                   'use_cosine_annealing': self.use_cosine_annealing,
#                   't_cur': int(K.get_value(self.t_cur)),
#                   'eta_t': int(K.get_value(self.eta_t)),
#                   'eta_min': int(K.get_value(self.eta_min)),
#                   'eta_max': int(K.get_value(self.eta_max)),
#                   'init_verbose': self.init_verbose,
#                   'epsilon': self.epsilon,
#                   'amsgrad': self.amsgrad}
#         base_config = super(AdamW, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
