#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:05:43 2020

@author: Vladislav
"""
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except:
    HAS_CUPY = False

from typing import List

import tqdm

class KuramotoFast:
    def __init__(self, n_nodes: int, n_oscillators: int, sampling_rate: int, k_list: List[float], weight_matrix: np.ndarray, 
                 frequency_spread: float, noise_scale: float=1.0, use_cuda: bool=True, use_tqdm: bool=True, node_frequencies=None,  **kwargs):  
        """
            Implentation of nested Kuramoto model. The model consists of N nodes each with M oscillators. Each pair of nodes is connected with directed weight given by weight_matrix.

            :param n_nodes: number of nodes in the model
            :param n_oscillators: number of oscillators in each 
            :param sampling_raet: update rate of the model
            :param k_list: list of K values (within node shift) of the model. Should have length equal to number of nodes.
            :param weight matrix: 2d matrix of node vs node connectivity weight. Should have N_nodes x N_nodes shape.
            :param frequency_spread: spread of frequencies within a node. Frequencies of oscillators are defined as linspace from centra_frequency - frequency_spread to central_frequency + frequency_spread
            :param noise_scale: sigma of noise.
            :param use_cuda: use GPU (cupy) to compute the model?

        """  
        self._check_parameters(n_nodes, k_list, weight_matrix)
        
        if use_cuda:
            if HAS_CUPY:
                self.xp = cp
            else:
                raise RuntimeError('use_cuda = True while cupy is not installed!')
        else:
            self.xp = np

        self.n_nodes = n_nodes
        self.n_oscillators = n_oscillators
        self.k_list = k_list
        self.noise_scale=2*np.pi*noise_scale/sampling_rate
        
        self.frequency_spread = frequency_spread
        self.node_frequencies = node_frequencies
        
        self.weight_matrix = self.xp.array(weight_matrix)
        self.xp.fill_diagonal(self.weight_matrix, 0)
        self.weight_matrix = (self.weight_matrix/sampling_rate).T.reshape(*self.weight_matrix.shape, 1)

        self.sampling_rate = sampling_rate
        self.use_cuda = use_cuda
        self.disable_tqdm = not(use_tqdm)
        
        self._init_parameters()
        self._preallocate()
            
    def _check_parameters(self, n_nodes: int, k_list: List[float], weight_matrix: np.ndarray):
        if len(k_list) != n_nodes:
            raise RuntimeError(f'Size of k_list ({len(k_list)}) is not equal to number of nodes ({n_nodes}).')
            
        if np.ndim(weight_matrix) != 2 or (weight_matrix.shape[0] != weight_matrix.shape[1]):
            raise RuntimeError(f'weight_matrix should be a 2d square matrix, got {weight_matrix.shape} shape.')

        if weight_matrix.shape[0] != n_nodes or weight_matrix.shape[1] != n_nodes:
            raise RuntimeError(f'weight matrix should be a 2d matrix of size N_nodes x N_nodes, got {weight_matrix.shape} shape')
    
    def _init_parameters(self):       
        # Central frequencies of each oscillators are evenly spaced values  in [central_frequency - frequency_spread; central_frequncy + frequency_spread]
        # Because we use a complex engine here, we need to convert frequencies given in Hz to a step on complex unit circle. 
        omegas = self.xp.zeros(shape=(self.n_nodes, self.n_oscillators))

        for idx, frequency in enumerate(self.node_frequencies):
            freq_lower = frequency - self.frequency_spread
            freq_upper = frequency + self.frequency_spread
            omegas[idx] = self.xp.linspace(freq_lower, freq_upper, num=self.n_oscillators)

        omegas += self.xp.random.uniform(-0.1, 0.1, size=omegas.shape)
        self.omegas =  self.xp.exp(1j * (omegas * 2 * np.pi / self.sampling_rate)) 

        # C is an average influence of other oscillators within a node.
        C = self.xp.array(self.k_list)/(self.n_oscillators * self.sampling_rate)
        self.shift_coeffs = C.reshape(-1,1)

        # Random initial phase;
        # Same as central frequencies we need to convert it to a point on complex unit circle. 
        thetas = self.xp.random.uniform(-np.pi, np.pi, size=omegas.shape)
        self.phases = self.xp.exp(1j*thetas)
        
        self._complex_dtype = self.xp.complex64
        self._float_dtype = self.xp.float32

    def _preallocate(self):
        n_nodes, n_osc = self.phases.shape
        
        self._phase_conj = self.xp.empty_like(self.phases)
        self._external_buffer = self.xp.empty((n_nodes, n_nodes, n_osc), dtype=self.phases.dtype)
        
    def _internal_step(self):
        # Internal dynamics is how oscillators within a node influence each other. It is computed as pairwise phase difference for oscillators within a node.
        # We want to comptue an oscillator vs oscillator phase difference within each node -> get N_nodes x N_osc x N_osc tensor
        # However, in this implementation we dont have weightes for oscillators. So we can use a simple trick to avoid pairwise comparison and reduce computational overhead.
        # Lets note that we take a sum along N_osc in pairwise phase diff tensor -> we can reduce it to pairwise diff of oscillator vs mean node phase
        # Therefore instead of doing O(N_nodes x N_osc x N_osc) we just need to do O(N_nodes x N_osc) + O(n_osc)!

        self.xp.multiply(self.phases, self._phase_conj.sum(axis=1, keepdims=True), out=self._phase_conj)
        self.xp.conj(self._phase_conj, out=self._phase_conj)
                
    def simulate(self, time: float, noise_realisations: int=100, random_seed: int=42) -> np.ndarray:
        """
            Implentation of nested Kuramoto model. The model consists of N nodes each with M oscillators. Each pair of nodes is connected with directed weight given by weight_matrix.

            :param time: Length of the simulation in seconds. Total number of samples is computed as sampling_rate x time + 1 (initial state)
            :param noise_realisations: Number of noise realisations to generate. 

            :return: N_nodes x N_ts matrix of complex values that contains each node activity during the simulation
        """
        xp = self.xp
        
        xp.random.seed(random_seed)

        n_iters = int(time*self.sampling_rate)
        history = xp.zeros((self.phases.shape[0], n_iters+1), dtype=self._complex_dtype)
        history[:, 0] = self.phases.mean(axis=1)
        
        # we want to generate noise before simulation & reuse it each iteration
        # it should save a lot of time because we do not need to generate it each iteration

        
        for i in tqdm.trange(1, n_iters+1, leave=False, desc='Kuramoto model is running...', disable=self.disable_tqdm):
            mean_phase = self.phases.mean(axis=1)
            xp.conj(self.phases, out=self._phase_conj)
            
            # External dynamics is how other nodes influence oscillators of a node. It is computed as phase difference of each oscillator with mean phase of each other node.
            # We want to compute an oscillator vs node phase difference for each node -> get N_nodes x N_nodes x N_osc tensor
            # Because we want the difference to be weighted we also need to multiply it on N_nodes x N_nodes weight matrix.
            self._external_buffer = xp.tensordot(self._phase_conj, mean_phase, axes=0).transpose(0,2,1)
            self._external_buffer *= self.weight_matrix
            
            external = self._external_buffer.sum(axis=1)
            
            # see comments in _internal_step function
            self._internal_step()

            # Working with imaginary part of complex number is the same as working with sin -> we dont need to convert it to angle and we can save a lot of computations
            internal = xp.exp(1j * xp.imag(self._phase_conj) * self.shift_coeffs)
            external = xp.exp(1j * xp.imag(external) / self.n_nodes)

            
            # Total phase shift is : natural dynamics (based on oscillator frequency) + internal dynamics + external dynamics
            phase_shift = self.omegas.copy()
            phase_shift *= internal 
            phase_shift *= external
            
            shift_noise = xp.random.normal(size=self.omegas.shape, loc=0, scale=self.noise_scale).astype(self._float_dtype)
            shift_noise = xp.exp(1j*shift_noise)

            # Add some noise to make model less linear and  prevent possible degradation to simple sin-like

            phase_shift *= shift_noise
            self.phases *= phase_shift
            
            history[:, i] = self.phases.mean(axis=1)
            
        if not(self.xp is np):
            history = self.xp.asnumpy(history)
    
        return history
    
class KuramotoFastWeighted(KuramotoFast):
    def __init__(self,  oscillator_weights: np.ndarray, **kwargs):
        """
            Implentation of nested Kuramoto model. The model consists of N nodes each with M oscillators. Each pair of nodes is connected with directed weight given by weight_matrix.

            :param n_nodes: number of nodes in the model
            :param n_oscillators: number of oscillators in each 
            :param sampling_raet: update rate of the model
            :param k_list: list of K values (within node shift) of the model. Should have length equal to number of nodes.
            :param weight matrix: 2d matrix of node vs node connectivity weight. Should have N_nodes x N_nodes shape.
            :param central_frequency: central frequency of the model.
            :param frequency_spread: spread of frequencies within a node. Frequencies of oscillators are defined as linspace from centra_frequency - frequency_spread to central_frequency + frequency_spread
            :param noise_scale: sigma of noise.
            :param oscillator_weights: internal weights of oscillators . Should be a 2d matrix of size N_oscillators x N_oscillators
            :param use_cuda: use GPU (cupy) to compute the model?

        """  

        super().__init__(**kwargs)

        self.osc_weights = self.xp.array(oscillator_weights)

            
    def _internal_step(self):
        # Internal dynamics is how oscillators within a node influence each other. It is computed as pairwise phase difference for oscillators within a node.
        # We want to comptue an oscillator vs oscillator phase difference within each node -> get N_nodes x N_osc x N_osc tensor
        # In this implementation each pair of oscillators has its own weight (based on central frequency difference or any other reason)
        # Therefore we cant simplify computations to O(N_nodes x N_osc) and have to compute all pairwise differences. 
        self._phase_conj = self.xp.einsum('ij,ik,jk->ik', self.phases, self._phase_conj, self.osc_weights, optimize=True)     