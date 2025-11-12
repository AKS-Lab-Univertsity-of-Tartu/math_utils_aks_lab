import os
from ament_index_python.packages import get_package_share_directory


xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


from functools import partial

import jax
import jax.numpy as jnp
import jax.nn
from jaxopt import linear_solve
import lineax as lx



class QP():

	def __init__(self, num_batch, num_dof, nvar, num_total_constraints, rho_ineq,
			  A_projection, A_control, A_eq, b_control, maxiter_projection):
		
		super(QP, self).__init__()
		
		self.A_projection = A_projection
		self.A_control = A_control
		self.A_eq = A_eq
		self.b_control = b_control

		self.rho_ineq = rho_ineq
		self.nvar = nvar
		self.num_batch = num_batch
		self.num_dof = num_dof

		self.num_total_constraints = num_total_constraints
		self.maxiter_projection = maxiter_projection

		self.compute_boundary_vec_batch = (jax.vmap(self.compute_boundary_vec_single, in_axes = (0)  )) # vmap parrallelization takes place over first axis
		self.cost_mat = self.compute_cost_mat()
		self.cost_mat_inv = jnp.linalg.pinv(self.cost_mat)
		# print("self.cost_mat_inv", self.cost_mat_inv)
		# print("self.cost_mat_inv shape", self.cost_mat_inv.shape)
		self.solve_batched_systems = jax.vmap(self.solve_single_system, in_axes=(0))

		eigvals, _ = jnp.linalg.eig(self.cost_mat)
		det_cost = jnp.linalg.det(self.cost_mat)

		# Store them if you want to inspect or reuse
		self.eigvals = eigvals
		# self.eigvecs = eigvecs
		self.det_cost = det_cost

		# jax.debug.print("Eigenvalues: {}", eigvals)
		# jax.debug.print("len(Eigenvalues): {}", len(eigvals))
		# jax.debug.print("max(Eigenvalues): {}", max(eigvals))
		# jax.debug.print("min(Eigenvalues): {}", min(eigvals))
		# jax.debug.print("Determinant: {}", det_cost)
    
	@partial(jax.jit, static_argnums=(0,))
	def get_rank(self, tol=1e-8):
		# Compute singular values
		s = jnp.linalg.svd(self.cost_mat, compute_uv=False)
		# Count number of singular values greater than tolerance
		rank = jnp.sum(s > tol)
		return rank
	
	@partial(jax.jit, static_argnums=(0,))
	def stable_logdet(self):
		s = jnp.linalg.svd(self.cost_mat, compute_uv=False)
		# log(det(A)) = sum(log(s_i))
		logdet = jnp.sum(jnp.log(s + 1e-20))  # avoid log(0)
		return logdet
	
	@partial(jax.jit, static_argnums=(0,))
	def get_svd(self):
		s = jnp.linalg.svd(self.cost_mat, compute_uv=False)
		# log(det(A)) = sum(log(s_i))
		
		return s

	
	@partial(jax.jit, static_argnums=(0,))
	def compute_cost_mat(self):
		# Cost matrix
		cost = (
			jnp.dot(self.A_projection.T, self.A_projection) +
			self.rho_ineq * jnp.dot(self.A_control.T, self.A_control)
		)
		# cost += 1e-6 * jnp.eye(cost.shape[0])

		# KKT system matrix
		cost_mat = jnp.vstack((
			jnp.hstack((cost, self.A_eq.T)),
			jnp.hstack((self.A_eq, jnp.zeros((self.A_eq.shape[0], self.A_eq.shape[0]))))
		))


		# jax.debug.print("cost_mat {}", jnp.shape(cost_mat))

		# return jnp.linalg.pinv(cost_mat)
		return cost_mat
	@partial(jax.jit, static_argnums=(0,))
	def solve_single_system(self, b_single):
		"""Solves the KKT system for a single sample."""
		# A is (186, 186), b_single is (186,)
		operator = lx.MatrixLinearOperator(self.cost_mat)
		
		# Use your desired iterative solver (e.g., lx.GMRES or lx.BiCGStab)
		solution = lx.linear_solve(operator, b_single, solver=lx.BiCGStab(rtol=1e-3, atol=1e-3, max_steps=20))
		# solution = lx.linear_solve(operator, b_single, solver=lx.GMRES(rtol=1e-6, atol=1e-6, max_steps=2000, restart=50))

		
		# NOTE: You'd replace lx.QR() with lx.GMRES() or lx.BiCGStab()
		
		return solution.value 
		



	@partial(jax.jit, static_argnums=(0,))
	def compute_boundary_vec_single(self, state_term):

		#FIX: Use integer division (// or jnp.floor_divide) 
        # to ensure num_eq_constraint is an integer for reshape()
		num_eq_constraint = jnp.shape(state_term)[0] // self.num_dof
		
		b_eq_term = state_term.reshape( num_eq_constraint,self.num_dof).T
		b_eq_term = b_eq_term.reshape(num_eq_constraint*self.num_dof)

		return b_eq_term
		
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_feasible_control(self, lamda_init, s_init, 
										 b_eq_term, xi_samples, 
										 init_pos):
		
	    
		def matvec_fun(x):
		
			return  jnp.dot(self.cost_mat, x)
		
		# Augmented bounds with slack variables
		b_control_aug = self.b_control - s_init

		# Linear cost term
		lincost = (
			-lamda_init -
			jnp.dot(self.A_projection.T, xi_samples.T).T -
			self.rho_ineq * jnp.dot(self.A_control.T, b_control_aug.T).T
		)

        


		# # Solve KKT system
		# sol = jnp.linalg.solve(self.cost_mat, jnp.hstack((-lincost, b_eq_term)).T).T
		# # # sol = (self.cost_mat_inv @ jnp.hstack((-lincost, b_eq_term)).T).T

		# # jax.debug.print("sol_1 {}", sol.shape)

		# sol =linear_solve.solve_normal_cg(matvec_fun, jnp.hstack((-lincost, b_eq_term)).T, tol=1e-5).T
		# # jax.debug.print("sol_2 {}", sol.shape)

		# operator = lx.MatrixLinearOperator(self.cost_mat)
		# # solution = lx.linear_solve(operator, vector, solver=lx.QR())

		# sol = lx.linear_solve(operator, jnp.hstack((-lincost, b_eq_term)).T, solver=lx.QR()).T

		#  Solve the batched system using the vmapped function
		B_matrix = jnp.hstack((-lincost, b_eq_term)).T 

        # CRITICAL FIX: Transpose B to put the 1000 vectors on the batch axis (axis 0)
		# b_batched = B_matrix.T  
		# sol_batched = self.solve_batched_systems(b_batched)
		# sol = sol_batched

		# sol =linear_solve.solve_normal_cg(matvec_fun, B_matrix, tol=1e-3).T

		# sol = jnp.linalg.solve(self.cost_mat, B_matrix).T
		
		sol = (self.cost_mat_inv @ B_matrix).T

		# sol =linear_solve.solve_normal_cg(matvec_fun, B_matrix, tol=1e-5).T




		# Extract primal solution
		xi_projected = sol[:, :self.nvar]

		# Update slack variables
		s = jnp.maximum(
			jnp.zeros((self.num_batch, self.num_total_constraints)),
			-jnp.dot(self.A_control, xi_projected.T).T + self.b_control
		)

		# s= jax.nn.relu(-jnp.dot(self.A_control, xi_projected.T).T + self.b_control)

		# s = jax.nn.leaky_relu(-jnp.dot(self.A_control, xi_projected.T).T + self.b_control, negative_slope=-0.001)

		# Compute residual
		res_vec = jnp.dot(self.A_control, xi_projected.T).T - self.b_control + s
		# res_norm = jnp.linalg.norm(res_vec, axis=1)
		# res_norm = jnp.square(res_vec, axis=1)
		res_norm = jnp.sum(res_vec**2, axis=1)
		
		lamda = lamda_init - self.rho_ineq * jnp.dot(self.A_control.T, res_vec.T).T



		# lamda = lamda_init - self.rho_ineq * jnp.dot(self.A_control.T, res_vec.T).T - mu*g_grads_filt

		return xi_projected, s, res_norm, lamda
	

	@partial(jax.jit, static_argnums=(0,))
	def compute_projection(self, xi_samples, state_term, lamda_init, 
						   s_init, init_pos):
		
		
		s_init = jnp.maximum(
			jnp.zeros((self.num_batch, self.num_total_constraints)),
			s_init
		)

		# s_init= jax.nn.relu(s_init)
		# s_init= jax.nn.leaky_relu(s_init, negative_slope=-0.001)
		
		b_eq_term = self.compute_boundary_vec_batch(state_term)  

		xi_projected_init = xi_samples

		def lax_custom_projection(carry, idx):
			_, lamda, s = carry
			lamda_prev, s_prev = lamda, s
			
			primal_sol, s, res_projection, lamda = self.compute_feasible_control(lamda, 
																		s, b_eq_term, xi_samples, 
																		init_pos)
			
			primal_residual = res_projection
			fixed_point_residual = (
				# jnp.linalg.norm(lamda_prev - lamda, axis=1) +
				# jnp.linalg.norm(s_prev - s, axis=1)

				jnp.sum((lamda_prev - lamda)**2, axis=1) +
				jnp.sum((s_prev - s)**2, axis=1)
			)
			return (primal_sol, lamda, s), (primal_residual, fixed_point_residual)

		carry_init = (xi_projected_init, lamda_init, s_init)


		carry_final, res_tot = jax.lax.scan(
			lax_custom_projection,
			carry_init,
			jnp.arange(self.maxiter_projection)
		)

		primal_sol, lamda, s = carry_final
		primal_residuals, fixed_point_residuals = res_tot

		primal_residuals = jnp.stack(primal_residuals)
		fixed_point_residuals = jnp.stack(fixed_point_residuals)

		return primal_sol, primal_residuals, fixed_point_residuals

	

	
	


    

	
	