# Optimization for fast linear averaging

import numpy as np
from cvxpy import *
import networkx as nx
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from graph_helper import * 
from test_graphs import * 
from diffusion import * 

def communication_costs(n, sensor_locations, cost_function):
	"""Returns a matrix of pairwise communication costs.

	Computes pairwise distances between robots and applies user-provided
	cost_function to these distances to obtain pairwise communication costs."""

	def proximity(i, j):
		dist = np.sum((sensor_locations[i, :] - sensor_locations[j, :]) ** 2) ** 0.5
		return cost_function(dist) 

	cost_matrix = np.zeros((n, n))
	for i in range(1, n):
		for j in range(0, i):  # note that diagonal entries must be 0
			cost = proximity(i, j)
			cost_matrix[i, j] = cost
			cost_matrix[j, i] = cost	
	return cost_matrix

def true_communication_cost(adjacency_matrix, cost_alpha):
	n = adjacency_matrix.shape[0]
	for i in range(n):
		adjacency_matrix[i, i] = 0.  # zero out the diagonals
	return np.sum(adjacency_matrix) * cost_alpha / n

def fast_linear_averaging(adjacency_matrix=None, cost_matrix=None, cost_alpha=0., tol=1e-4):
	"""
	Computes optimal Laplacian for symmetric fast linear averaging for a given graph 
	with given communication costs.

	Args:
		n: number of graph nodes
		adjacency_matrix: n x n symmetric adjacency adjacency matrix of 
			graph. If none, graph is assumed to be fully connected.
		cost_matrix: n x n symmetric matrix of communication costs. 
			Entries are assumed to be nonnegative, and diagonal entries must be 0.
			If None, communication costs are 0. 

	Returns:
		Optimal Laplacian 
	""" 

	n = adjacency_matrix.shape[0]

	L = Semidef(n)
	identity = np.identity(n)
	J = np.ones((n, n))
	gamma = Variable(1)
	convergence_matrix = identity - L - J/n 

	communication_cost = 0. if cost_matrix is None else cost_alpha * trace(cost_matrix * abs(-L)) / n 
	cost = gamma + communication_cost 

	constraints = [
	-gamma * identity << convergence_matrix,
	convergence_matrix << gamma * identity,
	L * np.ones((n, 1)) == np.zeros((n, 1))
	]

	# Constraints imposed by graph
	if adjacency_matrix is not None:
		for i in range(n):
			for j in range(n):
				if i != j and adjacency_matrix[i, j] == 0:
					constraints.append(L[i, j] == 0)

	objective = Minimize(cost)
	problem = Problem(objective, constraints)
	result = problem.solve()

	results_dict = {
		"status" : problem.status,
		"laplacian" : threshold_weights(L.value, tol),
		"lcp_matrix" : np.identity(n) - threshold_weights(L.value, tol),
		"spectral_gap" : gamma.value,
		"communication_cost" : 0. if cost_matrix is None else communication_cost.value,
		"total_cost" : problem.value
	}

	return results_dict

def distributed_leader_selection(adjacency_matrix, alpha=0.5, beta=0.25, kappa=0.5, mu=1.3):
	"""
		alpha: parameter for computing PPR-based importance scores (0.5 recommended)

		beta: parameter for computing PPR matrix that controls spread (still experimenting)
		kappa: regularization parameter. Larger kappa -> more concentration over x. 

		See paper (upcoming) for full details of the algorithm.
	"""
	n = adjacency_matrix.shape[0]
	x = Variable(n)
	I = np.identity(n)

	# First, compute the results of ordinary fast linear averaging
	flda_results = fast_linear_averaging(adjacency_matrix)
	lcp_matrix = flda_results["lcp_matrix"]
	
	# Use these results to compute a vector of scores related to PageRank. The dot product of these 
	# scores with x (the optimization variable) forms the part of our objective encouraing centrality.
	pagerank_vec = global_page_rank(lcp_matrix, 0.1)
	ppr_double_sos_vector = ppr_double_squared(lcp_matrix, alpha)
	alignment = n * ppr_double_sos_vector.T * x  # Want this less than 1

	# Compute the part of our objective that controls spread and sparsity
	Z = beta * np.linalg.inv(I - (1. - beta) * lcp_matrix)
	lambda1 = smallest_eigenvalue(Z)
	print("smallest_eigenvalue: ", lambda1)
	concentration = n * quad_form(x, Z - kappa * lambda1 * I)

	# Solve the optimization problem
	objective = Minimize(alignment + mu * concentration)
	constraints = [
	x >= 0,
	np.ones(n).T * x == 1.
	]
	problem = Problem(objective, constraints)
	problem.solve()

	results_dict = {
		"scaled_pagerank_vec" : n * pagerank_vec,
		"ppr_double_sos_vector" : n * ppr_double_sos_vector,
		"status" : problem.status,
		"x" : x.value,
		"scaled_x" : n * x.value,
		"objective" : problem.value,
		"alignment" : alignment.value,
		"concentration" : concentration.value
	}
	return results_dict

def dual_sdp_leader_selection(adjacency_matrix, min_leaders):
	"""Experimental. Solves dual problem for nonconvex variant of leader selection.

	The hope is that despite nonconvexity, dual optimal solution will be useful/a good 
	approximation to optimal solution. Needs more experimenting.
	"""
	# Basic setup/constants
	n = adjacency_matrix.shape[0]
	I = np.identity(n)
	ones_vec = np.ones(n)
	m = 1./min_leaders

	# Get lcp_matrix and form G_hat
	flda_results = fast_linear_averaging(adjacency_matrix)
	lcp_matrix = flda_results["lcp_matrix"]
	G_hat = np.matmul(lcp_matrix, lcp_matrix)
	for i in range(n):
		G_hat[i, i] = 0.

	# Dual problem setup
	b = Variable(n)
	lambda_ = Variable(1)
	nu = Variable(1)

	matrix = G_hat + lambda_ * I

	constraints = [matrix >> 0, b >= nu / 2.]
	objective = matrix_frac(b, matrix) + (m * lambda_ - nu)
	problem = Problem(Minimize(objective), constraints)
	problem.solve()

	# Recover x
	matrix_opt = matrix.value
	b_opt = b.value
	x = -np.linalg.pinv(matrix_opt).dot(b_opt)
	print("sum: ", np.sum(x))

	print("x: ", x)
	draw_graph_from_adjacency(adjacency_matrix)
	
def reoptimizer(n, cost_matrix, cost_alpha, tol):
	"""
	First solves relaxed convex optimization problem with
	cost_matrix, then uses solution returned to refine weights
	in a manner consistent with the returned topology.
	"""
	results_dict = fast_linear_averaging(adjacency_matrix=None, cost_matrix=cost_matrix, cost_alpha=cost_alpha)
	#print(results_dict)
	original_laplacian = results_dict["laplacian"]
	adjacency_matrix = laplacian_to_unweighted_adjacency(original_laplacian, tol)
	true_comm_cost = true_communication_cost(adjacency_matrix, cost_alpha)
	#print("True communication cost: ", true_comm_cost)
	reoptimized_results = fast_linear_averaging(adjacency_matrix=adjacency_matrix, cost_matrix=None, cost_alpha=0.)
	#print(reoptimized_results)
	return results_dict, reoptimized_results
