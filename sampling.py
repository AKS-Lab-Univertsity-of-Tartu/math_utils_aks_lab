import os


xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


from functools import partial
import numpy as np


import jax
import jax.numpy as jnp
import jax.nn as jnn


# # Get the folder containing this script
# current_dir = os.path.dirname(os.path.abspath(__file__))  # if in a script
# # current_dir = os.getcwd()  # if in Jupyter notebook

# # Add parent folder to sys.path
# parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)




class SAMPLING():

	def __init__(self, num_batch=None, nvar=None, lamda=None, num_elite=None, alpha_mean=None, alpha_cov=None):
		super(SAMPLING, self).__init__()
		
		self.num_batch = num_batch
		self.nvar = nvar
		self.lamda = lamda
		self.num_elite = num_elite
		self.alpha_mean = alpha_mean
		self.alpha_cov = alpha_cov
		self.vec_product = jax.jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))
		self.ellite_num = int(self.num_elite*self.num_batch)
	    

	
	@partial(jax.jit, static_argnums=(0, ))
	def compute_ellite_samples(self, cost_batch, xi_filtered):
		idx_ellite = jnp.argsort(cost_batch)
		cost_ellite = cost_batch[idx_ellite[0:self.ellite_num]]
		xi_ellite = xi_filtered[idx_ellite[0:self.ellite_num]]
		return xi_ellite, idx_ellite, cost_ellite
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_xi_samples(self, key, xi_mean, xi_cov ):
		key, subkey = jax.random.split(key)
		# xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov+0.009*jnp.identity(self.nvar), (self.num_batch, ))
		xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov, (self.num_batch, ))
		xi_samples = jnp.clip(xi_samples, a_min=-1.0, a_max=1.0)
		return xi_samples, key
	
	@partial(jax.jit, static_argnums=(0,))
	def comp_prod(self, diffs, d ):
		term_1 = jnp.expand_dims(diffs, axis = 1)
		term_2 = jnp.expand_dims(diffs, axis = 0)
		prods = d * jnp.outer(term_1,term_2)
		return prods	
	
	@partial(jax.jit, static_argnums=(0,))
	def repair_cov(self, C):
		epsilon = 1e-5
		eigenvalues, eigenvectors = jnp.linalg.eigh(C)
		min_eigenvalue = jnp.min(eigenvalues)
		def repair(_):
			clipped = jnp.where(eigenvalues < epsilon, epsilon, eigenvalues)
			D_prime = jnp.diag(clipped)
			C_repaired = eigenvectors @ D_prime @ eigenvectors.T
			# C_repaired = (C_repaired + C_repaired.T) / 2
			return C_repaired

		def keep(_):
			# cov_sym = (cov + cov.T) / 2
			return C

		C_repaired = jax.lax.cond(min_eigenvalue < epsilon, repair, keep, operand=None)
		return C_repaired
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_mean_cov(self, cost_ellite, mean_control_prev, cov_control_prev, xi_ellite):
		w = cost_ellite
		w_min = jnp.min(cost_ellite)
		w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
		sum_w = jnp.sum(w, axis = 0)
		mean_control = (1-self.alpha_mean)*mean_control_prev 
		mean_control += self.alpha_mean*(jnp.sum( (xi_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
		diffs = (xi_ellite - mean_control)
		prod_result = self.vec_product(diffs, w)
		cov_control = (1-self.alpha_cov)*cov_control_prev 
		cov_control += self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.01*jnp.identity(self.nvar)
		cov_control = self.repair_cov(cov_control)
		return mean_control, cov_control
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_adaptive_mean_cov(self,cost_ellite,mean_control_prev,sigma_diag_prev,xi_samples):
		eps = 1e-4
		# 2. Evaluate cost
		costs = cost_ellite  # (batch, num_samples)
		# 3. Soft weights
		weights = jnn.softmax(-costs * self.lamda)
		# 4. Update mean
		mu_new = jnp.sum(xi_samples * weights[:, None], axis=0)
		# mu_new = jnp.sum(weights_expanded * xi_samples, axis=1)
		# 5. Update diagonal covariance
		# diff = xi_samples - mu_new[:, None]
		diff = xi_samples - mu_new
		var_new = jnp.sum(weights[:, None] * (diff ** 2), axis=0)
		# var_new = jnp.sum(weights_expanded * (diff ** 2), axis=1)
		# 6. Stabilize
		# sigma_new = jnp.sqrt(var_new + eps)
		# sigma_diag = jnp.diag(sigma_new)
		sigma_new = var_new + eps 
		sigma_diag = jnp.diag(sigma_new)
		return mu_new, sigma_diag

