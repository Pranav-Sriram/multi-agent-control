
# Moved experimental code no longer used in core algorithms here.

import numpy as np
from cvxpy import *
from graph_helper import * 
from test_graphs import * 
from diffusion import * 

def asymmetric_fast_linear_averaging(n, adjacency_matrix=None):
	"""
	Computes optimal, possibly asymmetric LCP weights for fast linear averaging for a given graph.

	Given graph topology via the adjacency_matrix, computes the LCP weight matrix W that minimizes 
	the second largest eigenvalue magnitude of W, subject to the topological constraints imposed by adjacency_matrix.
	Note that in the asymmetric case, the SLEM of W is NOT equal to its spectral radius, unlike in the symmetric case.
	Hence we are not optimizing the asymptotic convergence rate but rather a proxy, the step-convergence rate.

	Args:
		n: number of graph nodes
		adjacency_matrix: n x n symmetric adjacency adjacency matrix of 
			graph. If none, graph is assumed to be fully connected.

	Returns:
		Optimal Laplacian 
	""" 

	W = Variable(n, n)  # possibly asymmetric, lcp weight matrix
	identity = np.identity(n)
	J = np.ones((n, n))
	gamma = Variable(1)
	convergence_matrix = W - J/n

	constraints = [
	-gamma * identity << convergence_matrix,
	convergence_matrix << gamma * identity,
	W * np.ones((n, 1)) == np.ones((n, 1)),  # row sums = 1
	np.ones((1, n)) * W == np.ones((1, n))  # col sums = 1
	]

	# Constraints imposed by graph
	if adjacency_matrix is not None:
		for i in range(n):
			for j in range(n):
				if i != j and adjacency_matrix[i, j] == 0:
					constraints.append(W[i, j] == 0)

	objective = Minimize(gamma)
	problem = Problem(objective, constraints)
	result = problem.solve()

	results_dict = {
		"status" : problem.status,
		"lcp_matrix" : W.value,
		"gamma" : gamma.value
	}

	return results_dict


# Electrical Resistances

def effective_resistance_weighting(adjacency_matrix, callback):
	"""Computes an lcp matrix based on edge effective resistances."""

	n = adjacency_matrix.shape[0]
	optimal_laplacian = fast_linear_averaging(n, adjacency_matrix)["laplacian"]
	optimal_lambda2 = second_smallest_eigenvalue(optimal_laplacian)

	laplacian = adjacency_to_laplacian(adjacency_matrix)

	effective_resistance_matrix = compute_effective_resistances(laplacian)
	masked_effective_resistance_matrix = effective_resistance_matrix * adjacency_matrix
	print("Masked ERM: ", masked_effective_resistance_matrix)

	transformed_matrix = callback(masked_effective_resistance_matrix)


	resistance_row_sums = np.sum(transformed_matrix, 1)
	max_row_sum = np.amax(resistance_row_sums)
	print("Max row sum: ", max_row_sum)

	# Normalize effective_resistance_matrix so rows sum to 1
	lcp_matrix = transformed_matrix / (max_row_sum + 1e-5)
	for i in range(n):
		lcp_matrix[i, i] = 1. - np.sum(lcp_matrix[i, :])
	output_laplacian = np.identity(n) - lcp_matrix
	print(lcp_matrix)

	print("Final lambda2: ", second_smallest_eigenvalue(output_laplacian))
	print("Optimal lambda2: ", optimal_lambda2)

	print("Final lambda_max: ", largest_eigenvalue(output_laplacian))
	print("Optimal lambda_max: ", largest_eigenvalue(optimal_laplacian))

	#print("Spectrum of optimal laplacian: ", spectrum(optimal_laplacian))
	#print("Spectrum of output laplacian: ", spectrum(output_laplacian))

	draw_graph_from_adjacency(adjacency_matrix)
	draw_graph_from_laplacian(output_laplacian)
	draw_graph_from_laplacian(optimal_laplacian)