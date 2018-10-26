import numpy as np 
from graph_helper import * 

# Experimental work on the following question: can we learn good diffusion weights on a 
# graph in an unsupervised, iterative, distributed way? We're interested in the same types of optimization
# problems as in eigenvalue_optimization, but now seek iterative algorithms that rely solely on local 
# information.

def basic_diffusion_test(adjacency_matrix, iters):
	num_robots = adjacency_matrix.shape[0]
	initial_vals = np.zeros(num_robots)
	initial_vals[0] = 1.
	final_vals = diffusion_process_simulation(
		laplacian=naive_averaging_laplacian(adjacency_matrix),
		initial_node_vals=initial_vals,
		iters=iters)
	print("Final vals: ", final_vals)


def ppr_iteration(init_diffusion_matrix, alpha, iters, lr=1e-1, tol=1e-4):
	diffusion_matrix = np.copy(init_diffusion_matrix)
	n = diffusion_matrix.shape[0]
	for it in range(iters):
		if it % 200 == 0:
			print("finished iter ", it)
			print("current lambda2: ", second_smallest_eigenvalue(np.identity(n) - diffusion_matrix))
		for i in range(n):
			ppr_vec = personalized_page_rank(
				diffusion_matrix, alpha, seed_vector=build_seed_vector([i], n))
			mask = np.where(np.abs(init_diffusion_matrix[i, :]) > tol, np.ones(n), np.zeros(n))
			degree = np.sum(mask)

			# invert->mask->normalize->update->symmetrize->maintain
			#inverse_ranks = 1. / (ppr_vec + 1e-9)
			ppr_vec *= mask  # mask out non-neighbors (but keep self-edge)
			ppr_vec /= (np.sum(ppr_vec) + 1e-7) # normalize
			new_vals = (2./degree * mask) - ppr_vec
			
			for j in range(n):
				if mask[j] > 0:
					new_val = new_vals[j]
					delta = new_val - diffusion_matrix[i, j]  # new - old
					diffusion_matrix[i, j] += delta  # update
					if j != i:
						diffusion_matrix[j, i] += delta  # symmetrize
						diffusion_matrix[j, j] -= delta  # maintain row-sum invariant
	return diffusion_matrix


def diffusion_iteration(adjacency_matrix, global_iters=5, iters_per_round=6, beta=0.02):
	num_robots = adjacency_matrix.shape[0]
	initial_laplacian = naive_averaging_laplacian(adjacency_matrix)
	print("Initial lambda2: ", second_smallest_eigenvalue(initial_laplacian))
	current_diffusion_matrix = np.identity(num_robots) - initial_laplacian

	def resymmetrize(current_diffusion_matrix, i, desired_minus_observed, beta):
		for j in range(num_robots):
			if adjacency_matrix[i, j] != 0:
				old_ival = current_diffusion_matrix[i, j]
				old_j_self_weight = current_diffusion_matrix[j, j]

				delta = beta * desired_minus_observed[j]
				new_delta = delta 

				if delta > old_j_self_weight:
					new_delta = old_j_self_weight  # clamp
				if old_j_self_weight - delta > 0.99: 
					new_delta = old_j_self_weight - 0.99
				i_absorption = delta - new_delta

				current_diffusion_matrix[i, j] += new_delta
				current_diffusion_matrix[j, i] += new_delta
				current_diffusion_matrix[j, j] -= new_delta
				current_diffusion_matrix[i, i] += i_absorption

		print(current_diffusion_matrix)  # test!
		return current_diffusion_matrix

	def update_diffusion_matrix(current_diffusion_matrix, final_vals, i, beta=0.1):
		neighbor_mask = adjacency_matrix[i, :]
		degree_i = np.sum(neighbor_mask)
		neighbor_mask[i] = 1.  # count self as neighbor for purpose of this algorithm
		neighbor_final_vals = neighbor_mask * final_vals  # element-wise masking; only these vals visible to node i

		if np.sum(neighbor_final_vals) > 0:
			normalized_neighbor_final_vals = neighbor_final_vals / np.sum(neighbor_final_vals)
		else:
			normalized_neighbor_final_vals = np.zeros(num_robots)

		#i_weights = current_diffusion_matrix[i, :]
		desired_minus_observed = 1. / (degree_i + 1.) - normalized_neighbor_final_vals
		desired_minus_observed *= neighbor_mask  # Make sure to zero out non-neighbors!

		return resymmetrize(current_diffusion_matrix, i, desired_minus_observed, beta)

	for global_iter in range(global_iters):
		curr_laplacian = np.identity(num_robots) - current_diffusion_matrix
		print("Current lambda2: ", second_smallest_eigenvalue(curr_laplacian))
		print("Current lambda_max: ", largest_eigenvalue(curr_laplacian))

		for i in range(num_robots):  # simulate diffusion from i and update diffusion matrix
			initial_vals = np.zeros(num_robots)
			initial_vals[i] = 1.  # seed node
			final_vals = diffusion_process_simulation(
				current_diffusion_matrix, initial_vals, iters_per_round)
			current_diffusion_matrix = update_diffusion_matrix(
				current_diffusion_matrix, final_vals, i, beta)

	final_laplacian = np.identity(num_robots) - current_diffusion_matrix
	final_lambda2 = second_smallest_eigenvalue(final_laplacian)
	print("Final lambda2: ", final_lambda2)

	return final_laplacian, current_diffusion_matrix