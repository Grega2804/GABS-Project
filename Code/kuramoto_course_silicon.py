#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import List
import tqdm

# Try to import MLX for Apple Silicon Acceleration
try:
    import mlx.core as mx
    HAS_ACCEL = True
except ImportError:
    HAS_ACCEL = False
    print("NO ACCELERATION!")

class KuramotoFast:
    def __init__(self, n_nodes: int, n_oscillators: int, sampling_rate: int, k_list: List[float], 
                 weight_matrix: np.ndarray, frequency_spread: float, noise_scale: float=1.0, 
                 use_accel: bool=True, use_tqdm: bool=True, node_frequencies=None, **kwargs):  
        
        self._check_parameters(n_nodes, k_list, weight_matrix)
        
        if use_accel and HAS_ACCEL:
            self.xp = mx
            self.is_mlx = True
        else:
            self.xp = np
            self.is_mlx = False

        self.n_nodes = n_nodes
        self.n_oscillators = n_oscillators
        self.k_list = k_list
        self.noise_scale = 2 * np.pi * noise_scale / sampling_rate
        self.frequency_spread = frequency_spread
        self.node_frequencies = node_frequencies
        
        self.weight_matrix = self.xp.array(weight_matrix)
        
        if self.is_mlx:
            mask = (1 - mx.eye(n_nodes)).astype(self.weight_matrix.dtype)
            self.weight_matrix = self.weight_matrix * mask
        else:
            np.fill_diagonal(self.weight_matrix, 0)

        self.weight_matrix = (self.weight_matrix / sampling_rate).T.reshape(*self.weight_matrix.shape, 1)

        self.sampling_rate = sampling_rate
        self.disable_tqdm = not(use_tqdm)
        
        self._init_parameters()
        
    def _check_parameters(self, n_nodes, k_list, weight_matrix):
        if len(k_list) != n_nodes or np.ndim(weight_matrix) != 2 or weight_matrix.shape[0] != weight_matrix.shape[1]:
            raise RuntimeError('Invalid parameters.')

    def _init_parameters(self):       
        omegas = self.xp.zeros(shape=(self.n_nodes, self.n_oscillators))
        for idx, frequency in enumerate(self.node_frequencies):
            freq_lower = frequency - self.frequency_spread
            freq_upper = frequency + self.frequency_spread
            omegas[idx] = self.xp.linspace(freq_lower, freq_upper, num=self.n_oscillators)

        # --- FIX 1: Conditional shape/size for omegas ---
        if self.is_mlx:
            omegas += self.xp.random.uniform(-0.1, 0.1, shape=omegas.shape)
        else:
            omegas += self.xp.random.uniform(-0.1, 0.1, size=omegas.shape)
            
        self.omegas = self.xp.exp(1j * (omegas * 2 * np.pi / self.sampling_rate)) 

        C = self.xp.array(self.k_list) / (self.n_oscillators * self.sampling_rate)
        self.shift_coeffs = C.reshape(-1, 1)

        # --- FIX 2: Conditional shape/size for thetas ---
        if self.is_mlx:
            thetas = self.xp.random.uniform(-np.pi, np.pi, shape=omegas.shape)
        else:
            thetas = self.xp.random.uniform(-np.pi, np.pi, size=omegas.shape)
            
        self.phases = self.xp.exp(1j * thetas)
        
        self._complex_dtype = self.xp.complex64 if hasattr(self.xp, 'complex64') else np.complex64
        self._float_dtype = self.xp.float32 if hasattr(self.xp, 'float32') else np.float32


    def _create_compiled_step(self):
        """Fuses the entire mathematical step into a single optimized GPU kernel."""
        def step(phases):
            # External
            mean_phase = phases.mean(axis=1)
            p_conj = phases.conj()
            ext_buf = mx.tensordot(p_conj, mean_phase, axes=0).transpose(0, 2, 1)
            ext_buf = ext_buf * self.weight_matrix
            external_sum = ext_buf.sum(axis=1)
            
            # Internal
            sum_conj = p_conj.sum(axis=1, keepdims=True)
            phase_conj = (phases * sum_conj).conj()
            
            # Shifts
            internal = mx.exp(1j * mx.imag(phase_conj) * self.shift_coeffs)
            external = mx.exp(1j * mx.imag(external_sum) / self.n_nodes)
            phase_shift = self.omegas * internal * external
            
            # --- FIX 3: Use 'shape' instead of 'size' for MLX normal distribution ---
            shift_noise = mx.random.normal(shape=self.omegas.shape, loc=0, scale=self.noise_scale).astype(self._float_dtype)
            
            # Update phases
            new_phases = phases * phase_shift * mx.exp(1j * shift_noise)
            return new_phases, new_phases.mean(axis=1)
            
        return mx.compile(step)

    def simulate(self, time: float, noise_realisations: int=100, random_seed: int=42) -> np.ndarray:
        if self.is_mlx: mx.random.seed(random_seed)
        else: np.random.seed(random_seed)

        n_iters = int(time * self.sampling_rate)
        
        initial_mean = self.phases.mean(axis=1)
        
        if self.is_mlx:
            mx.eval(initial_mean)
            compiled_step = self._create_compiled_step()
            
            # Start history list with the initial 2D block
            history_list = [np.array(initial_mean).reshape(-1, 1)]
            
            chunk_size = 4000  # <--- HUGE boost to keep the GPU fed
            current_chunk = []
            
            for i in tqdm.trange(1, n_iters + 1, leave=False, desc='Running Kuramoto...', disable=self.disable_tqdm):
                self.phases, step_mean = compiled_step(self.phases)
                current_chunk.append(step_mean)
                
                # Evaluate and transfer in BULK
                if i % chunk_size == 0:
                    # 1. Stack the chunk into a single matrix directly on the GPU
                    chunk_stacked = mx.stack(current_chunk, axis=1)
                    # 2. Force evaluation of the graph
                    mx.eval(self.phases, chunk_stacked)
                    # 3. Do ONE single bulk memory transfer to Python/NumPy
                    history_list.append(np.array(chunk_stacked))
                    current_chunk = [] 
            
            # Catch any remaining steps
            if current_chunk:
                chunk_stacked = mx.stack(current_chunk, axis=1)
                mx.eval(self.phases, chunk_stacked)
                history_list.append(np.array(chunk_stacked))
                
            # Stitch the big blocks together
            history = np.concatenate(history_list, axis=1)
            
        else:
            # Standard CPU fallback loop
            history_list = [initial_mean]
            history = np.zeros((self.phases.shape[0], n_iters + 1), dtype=np.complex64)
            history[:, 0] = initial_mean
            xp = self.xp
            for i in tqdm.trange(1, n_iters + 1, leave=False, desc='Running Kuramoto CPU...', disable=self.disable_tqdm):
                mean_phase = self.phases.mean(axis=1)
                p_conj = self.phases.conj()
                ext_buf = xp.tensordot(p_conj, mean_phase, axes=0).transpose(0, 2, 1)
                ext_buf *= self.weight_matrix
                external_sum = ext_buf.sum(axis=1)
                sum_conj = p_conj.sum(axis=1, keepdims=True)
                phase_conj = (self.phases * sum_conj).conj()
                internal = xp.exp(1j * xp.imag(phase_conj) * self.shift_coeffs)
                external = xp.exp(1j * xp.imag(external_sum) / self.n_nodes)
                phase_shift = self.omegas * internal * external
                shift_noise = xp.random.normal(size=self.omegas.shape, loc=0, scale=self.noise_scale).astype(self._float_dtype)
                self.phases *= (phase_shift * xp.exp(1j * shift_noise))
                history[:, i] = self.phases.mean(axis=1)

        return history
    
class KuramotoFastWeighted(KuramotoFast):
    def __init__(self, oscillator_weights: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.osc_weights = self.xp.array(oscillator_weights)

    def _internal_step(self):
        # einsum is supported in both MLX and NumPy
        self._phase_conj = self.xp.einsum('ij,ik,jk->ik', self.phases, self.phases.conj(), self.osc_weights)