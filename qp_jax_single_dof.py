import os


xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


from functools import partial



import jax
import jax.numpy as jnp



class QP():

	def __init__(self, num_batch, num_dof, nvar_single, num_total_constraints_per_dof, rho_ineq,
			  A_projection_single_dof, A_control_single_dof, A_eq_single_dof,
			  v_max, a_max, j_max, p_max, num_pos_constraints, num_vel_constraints,
			  num_acc_constraints, num_jerk_constraints, maxiter_projection):
		super(QP, self).__init__()
		
		self.A_projection_single_dof = A_projection_single_dof
		self.A_control_single_dof = A_control_single_dof
		self.A_eq_single_dof = A_eq_single_dof
		self.rho_ineq = rho_ineq
		self.nvar_single = nvar_single
		self.num_batch = num_batch
		self.num_dof = num_dof

		self.num_total_constraints_per_dof = num_total_constraints_per_dof
		self.v_max = v_max
		self.a_max = a_max
		self.j_max = j_max
		self.p_max = p_max
		self.num_pos_constraints = num_pos_constraints
		self.num_vel_constraints = num_vel_constraints
		self.num_acc_constraints = num_acc_constraints
		self.num_jerk_constraints = num_jerk_constraints
		self.maxiter_projection = maxiter_projection
		
		
		self.compute_boundary_vec_batch_single_dof = (jax.vmap(self.compute_boundary_vec_single_dof, in_axes = (0)  )) # vmap parrallelization takes place over first axis

	
	@partial(jax.jit, static_argnums=(0,))
	def compute_boundary_vec_single_dof(self, state_term):
		num_eq_constraint_per_dof = int(jnp.shape(state_term)[0])
		b_eq_term = state_term.reshape( num_eq_constraint_per_dof).T
		b_eq_term = b_eq_term.reshape(num_eq_constraint_per_dof)
		return b_eq_term
	

	@partial(jax.jit, static_argnums=(0,))
	def compute_feasible_control_single_dof(self, lamda_init_single_dof, s_init_single_dof, 
										 b_eq_term_single_dof, xi_samples_single_dof, 
										 init_pos_single_dof):
		b_vel = jnp.hstack((
			self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // (2*self.num_dof))),
			self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // (2*self.num_dof)))
		))

		b_acc = jnp.hstack((
			self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // (2*self.num_dof))),
			self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // (2*self.num_dof)))
		))

		b_jerk = jnp.hstack((
			self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // (2*self.num_dof))),
			self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // (2*self.num_dof)))
		))
        

		init_pos_single_dof_batch = jnp.tile(init_pos_single_dof, (self.num_batch, 1))  # (num_batch, 1)
        
		# Calculate bounds for each joint and each batch
    	# Upper bounds: p_max - init_pos, Lower bounds: p_max + init_pos (assuming symmetric limits)
		b_pos_upper = (self.p_max - init_pos_single_dof_batch)  # shape (num_batch, 1)
		b_pos_lower = (self.p_max + init_pos_single_dof_batch)  # shape (num_batch, 1)
        
		
		# Expand to include time steps
		b_pos_upper_expanded = jnp.tile(b_pos_upper[:, :, None], (1, 1, self.num_pos_constraints // (self.num_dof * 2)))  # (num_batch, 1, num_pos_constraints per dof/2)
		b_pos_lower_expanded = jnp.tile(b_pos_lower[:, :, None], (1, 1, self.num_pos_constraints // (self.num_dof * 2)))  # (num_batch, 1, num_pos_constraintsper dof/2)
		
		# Stack upper and lower bounds
		b_pos_stacked = jnp.concatenate([b_pos_upper_expanded, b_pos_lower_expanded], axis=2)  # (num_batch, 1, num_pos_constraints per dof)
		
		# Reshape to final form: (num_batch, total_pos_constraints)
		b_pos = b_pos_stacked.reshape((self.num_batch, -1))  # shape: (num_batch, self.num_pos_constraints per dof)
        
		b_control_single_dof = jnp.hstack((b_vel, b_acc, b_jerk, b_pos))
		
		s_init_single_dof = jnp.maximum(
			jnp.zeros((self.num_batch, self.num_total_constraints_per_dof)),
			s_init_single_dof
		)
	

		# Augmented bounds with slack variables
		b_control_aug_single_dof = b_control_single_dof - s_init_single_dof

		# Cost matrix
		cost = (
			jnp.dot(self.A_projection_single_dof.T, self.A_projection_single_dof) +
			self.rho_ineq * jnp.dot(self.A_control_single_dof.T, self.A_control_single_dof)
		)

		# KKT system matrix
		cost_mat = jnp.vstack((
			jnp.hstack((cost, self.A_eq_single_dof.T)),
			jnp.hstack((self.A_eq_single_dof, jnp.zeros((self.A_eq_single_dof.shape[0], self.A_eq_single_dof.shape[0]))))
		))

		# Linear cost term
		lincost = (
			-lamda_init_single_dof -
			jnp.dot(self.A_projection_single_dof.T, xi_samples_single_dof.T).T -
			self.rho_ineq * jnp.dot(self.A_control_single_dof.T, b_control_aug_single_dof.T).T
		)

		# Solve KKT system
		sol = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost, b_eq_term_single_dof)).T).T

		# Extract primal solution
		xi_projected = sol[:, :self.nvar_single]

		# Update slack variables
		s = jnp.maximum(
			jnp.zeros((self.num_batch, self.num_total_constraints_per_dof)),
			-jnp.dot(self.A_control_single_dof, xi_projected.T).T + b_control_single_dof
		)

		# Compute residual
		res_vec = jnp.dot(self.A_control_single_dof, xi_projected.T).T - b_control_single_dof + s
		res_norm = jnp.linalg.norm(res_vec, axis=1)

		# Update Lagrange multipliers
		lamda = lamda_init_single_dof - self.rho_ineq * jnp.dot(self.A_control_single_dof.T, res_vec.T).T

		return xi_projected, s, res_norm, lamda
	

	@partial(jax.jit, static_argnums=(0,))
	def compute_projection_single_dof(self, 
								       xi_samples_single_dof, 
								       state_term_single_dof, 
									   lamda_init_single_dof, 
									   s_init_single_dof, 
									   init_pos_single_dof):
		
		
		# state_term_single_dof: (B, K) → flatten across batch
		b_eq_term = self.compute_boundary_vec_batch_single_dof(state_term_single_dof)  # should become (B, K), flattened
		

		xi_projected_init_single_dof = xi_samples_single_dof

		def lax_custom_projection(carry, idx):
			_, lamda, s = carry
			lamda_prev, s_prev = lamda, s
			
			primal_sol, s, res_projection, lamda = self.compute_feasible_control_single_dof(lamda, 
																		s, b_eq_term, xi_samples_single_dof, 
																		init_pos_single_dof)
			
			primal_residual = res_projection
			fixed_point_residual = (
				jnp.linalg.norm(lamda_prev - lamda, axis=1) +
				jnp.linalg.norm(s_prev - s, axis=1)
			)
			return (primal_sol, lamda, s), (primal_residual, fixed_point_residual)

		carry_init = (xi_projected_init_single_dof, lamda_init_single_dof, s_init_single_dof)


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

	

	
	


    

	
	