import os
from ament_index_python.packages import get_package_share_directory


xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


from functools import partial
import numpy as np

import mujoco
import mujoco.mjx as mjx 
import jax
import jax.numpy as jnp



class QP():

	def __init__(self):
		super(QP, self).__init__()
	 
		# self.num_dof = num_dof
		# self.num_batch = num_batch
		# self.t = timestep
		# self.num = num_steps
		# self.num_elite = num_elite

		# self.t_fin = self.num*self.t
		# self.init_joint_position = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])
		
		# tot_time = np.linspace(0, self.t_fin, self.num)
		# self.tot_time = tot_time
		# tot_time_copy = tot_time.reshape(self.num, 1)

		# self.P = jnp.identity(self.num) # Velocity mapping 
		# self.Pdot = jnp.diff(self.P, axis=0)/self.t # Accelaration mapping
		# self.Pddot = jnp.diff(self.Pdot, axis=0)/self.t # Jerk mapping
		# self.Pint = jnp.cumsum(self.P, axis=0)*self.t # Position mapping
		# self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
		# self.Pint_jax = jnp.asarray(self.Pint)

		# self.nvar_single = jnp.shape(self.P_jax)[1]
		# self.nvar = self.nvar_single*self.num_dof 
  
		# self.rho_ineq = 5.0
		# self.rho_projection = 1.0

		# self.A_projection_single_dof = jnp.identity(self.nvar_single)

		# A_v_ineq_single_dof, A_v_single_dof = self.get_A_v_single_dof()
		# self.A_v_ineq_single_dof = jnp.asarray(A_v_ineq_single_dof) 
		# self.A_v_single_dof = jnp.asarray(A_v_single_dof)

		# A_a_ineq_single_dof, A_a_single_dof = self.get_A_a_single_dof()
		# self.A_a_ineq_single_dof = jnp.asarray(A_a_ineq_single_dof) 
		# self.A_a_single_dof = jnp.asarray(A_a_single_dof)

		# A_j_ineq_single_dof, A_j_single_dof = self.get_A_j_single_dof()
		# self.A_j_ineq_single_dof = jnp.asarray(A_j_ineq_single_dof)
		# self.A_j_single_dof = jnp.asarray(A_j_single_dof)
  
		# A_p_ineq_single_dof, A_p_single_dof = self.get_A_p_single_dof()
		# self.A_p_ineq_single_dof = jnp.asarray(A_p_ineq_single_dof) 
		# self.A_p_single_dof = jnp.asarray(A_p_single_dof)

		# # Combined control matrix (like A_control in )
		# self.A_control_single_dof = jnp.vstack((
		# 	self.A_v_ineq_single_dof,
		# 	self.A_a_ineq_single_dof,
		# 	self.A_j_ineq_single_dof,
		# 	self.A_p_ineq_single_dof
		# ))

		# A_eq_single_dof = self.get_A_eq_single_dof()
		# self.A_eq_single_dof = jnp.asarray(A_eq_single_dof)

		# A_theta, A_thetadot, A_thetaddot, A_thetadddot = self.get_A_traj()

		# self.A_theta = np.asarray(A_theta)
		# self.A_thetadot = np.asarray(A_thetadot)
		# self.A_thetaddot = np.asarray(A_thetaddot)
		# self.A_thetadddot = np.asarray(A_thetadddot)
		
		# self.key= jax.random.PRNGKey(42)
		# self.maxiter_projection = maxiter_projection
		# self.maxiter_cem = maxiter_cem

		# self.v_max = max_joint_vel
		# self.a_max = max_joint_acc
		# self.j_max = max_joint_jerk
		# self.p_max = max_joint_pos		
		    
    	# # Calculating number of Inequality constraints
		# self.num_vel = self.num
		# self.num_acc = self.num - 1
		# self.num_jerk = self.num - 2
		# self.num_pos = self.num

		# self.num_vel_constraints = 2 * self.num_vel * num_dof
		# self.num_acc_constraints = 2 * self.num_acc * num_dof
		# self.num_jerk_constraints = 2 * self.num_jerk * num_dof
		# self.num_pos_constraints = 2 * self.num_pos * num_dof
		# self.num_total_constraints = (self.num_vel_constraints + self.num_acc_constraints + self.num_jerk_constraints + self.num_pos_constraints)
		# self.num_total_constraints_per_dof = 2*(self.num_vel + self.num_acc + self.num_jerk + self.num_pos)



		# self.compute_boundary_vec_batch_single_dof = (jax.vmap(self.compute_boundary_vec_single_dof, in_axes = (0)  )) # vmap parrallelization takes place over first axis
		# self.compute_projection_batched_over_dof = jax.vmap(self.compute_projection_single_dof, in_axes=(0, 0, 0, 0, 0)) # vmap parrallelization takes place over first axis

		# self.print_info()

	# def get_A_traj(self):

	# 	# This is valid while dealing with knots anfd projecting into pos,vel,acc space with Bernstein Polynomials
	# 	# A_theta = np.kron(np.identity(self.num_dof), self.P )
	# 	# A_thetadot = np.kron(np.identity(self.num_dof), self.Pdot )
	# 	# A_thetaddot = np.kron(np.identity(self.num_dof), self.Pddot )
        
    #     # This is valid while not using knots and bernstein polynomials; directlly using velocity
	# 	A_theta = np.kron(np.identity(self.num_dof), self.Pint )
	# 	A_thetadot = np.kron(np.identity(self.num_dof), self.P )
	# 	A_thetaddot = np.kron(np.identity(self.num_dof), self.Pdot )
	# 	A_thetadddot = np.kron(np.identity(self.num_dof), self.Pddot )

	# 	return A_theta, A_thetadot, A_thetaddot, A_thetadddot	


	# def get_A_p_single_dof(self):
	# 	A_p = np.vstack(( self.Pint, -self.Pint))
	# 	A_p_ineq = np.kron(np.identity(1), A_p )
	# 	return A_p_ineq, A_p
	
	# def get_A_v_single_dof(self):
	# 	A_v = np.vstack(( self.P, -self.P     ))
	# 	A_v_ineq = np.kron(np.identity(1), A_v )
	# 	return A_v_ineq, A_v

	# def get_A_a_single_dof(self):
	# 	A_a = np.vstack(( self.Pdot, -self.Pdot  ))
	# 	A_a_ineq = np.kron(np.identity(1), A_a )
	# 	return A_a_ineq, A_a
	
	# def get_A_j_single_dof(self):
	# 	A_j = np.vstack(( self.Pddot, -self.Pddot  ))
	# 	A_j_ineq = np.kron(np.identity(1), A_j )
	# 	return A_j_ineq, A_j
	
	# def get_A_eq_single_dof(self):
	# 	return np.kron(np.identity(1), self.P[0])

	
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

	

	
	


    

	
	