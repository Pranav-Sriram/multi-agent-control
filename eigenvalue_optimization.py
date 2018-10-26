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

		beta: parameter for computing PPR matrix that controls spread (0.5 or greater recommended)
		kappa: regularization parameter. Larger kappa -> more concentration over x. 
	"""
	n = adjacency_matrix.shape[0]
	x = Variable(n)
	I = np.identity(n)

	flda_results = fast_linear_averaging(adjacency_matrix)
	lcp_matrix = flda_results["lcp_matrix"]


	#rw_matrix = adjacency_matrix / np.sum(adjacency_matrix, 1)
	#W_hat = 0.5 * (rw_matrix + I)
	
	pagerank_vec = global_page_rank(lcp_matrix, 0.1)
	ppr_double_sos_vector = ppr_double_squared(lcp_matrix, alpha)

	#ppr_matrix = beta * np.linalg.inv(np.identity(n) - (1. - beta) * lcp_matrix)

	#G = np.matmul(ppr_matrix, ppr_matrix)  # gram matrix 
	
	# print("ppr scores: ", np.diagonal(G))

	alignment = n * ppr_double_sos_vector.T * x  # Want this less than 1
	Z = beta * np.linalg.inv(I - (1. - beta) * lcp_matrix)
	lambda1 = smallest_eigenvalue(Z)
	#print("spectrum: ", spectrum(Z))
	print("smallest_eigenvalue: ", lambda1)

	concentration = n * quad_form(x, Z - kappa * lambda1 * I)
	objective = Minimize(alignment + mu * concentration)

	#objective = Maximize(alignment - kappa * concentration)

	
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


# Testing

def lambda_versus_alpha_test(n, cost_matrix, tol=0.01, alpha_vals=None):
	if alpha_vals is None:
		alpha_vals = np.arange(0.01, 0.5, 0.04)
	lambda_vals = np.zeros(alpha_vals.shape[0])
	for i, alpha in enumerate(alpha_vals):
		print("alpha: ", alpha)
		results_dict, reoptimized_results = reoptimizer(n, cost_matrix, alpha, tol)
		lambda_vals[i] = results_dict["spectral_gap"]
	plt.plot(alpha_vals, 1.-lambda_vals)
	plt.xlabel("α")
	plt.ylabel("λ_2")
	print(lambda_vals)
	plt.show()


def test(adjacency_matrix, print_full=False):
	num_robots = adjacency_matrix.shape[0]
	print("Adjacency matrix: ", adjacency_matrix)

	initial_laplacian = naive_averaging_laplacian(adjacency_matrix)
	initial_lambda2 = second_smallest_eigenvalue(initial_laplacian)
	initial_resistances = compute_effective_resistances(initial_laplacian)
	print("initial lambda2: ", initial_lambda2)
	if print_full: print("initial resistances: ", initial_resistances)
	print("initial total resistance: ", np.sum(initial_resistances) / 2.)

	positions = draw_graph_from_adjacency(adjacency_matrix)
	results = fast_linear_averaging(adjacency_matrix=adjacency_matrix)
	output_laplacian = threshold_weights(results["laplacian"], tol=5e-3)
	lcp_matrix = np.identity(num_robots) - output_laplacian
	print("Output LCP matrix: ", lcp_matrix)

	resistance_matrix = compute_effective_resistances(output_laplacian)
	final_lambda2 = second_smallest_eigenvalue(output_laplacian)
	print("Final lambda2: ", final_lambda2)
	if print_full: print("Resistances: ", resistance_matrix)
	print("Total final resistance: ", np.sum(resistance_matrix) / 2.)

	draw_graph_from_laplacian(output_laplacian, positions=positions)



def asymmetric_reff_weighting(adjacency_matrix):
	n = adjacency_matrix.shape[0]
	J = np.ones((n, n))
	optimal_laplacian = fast_linear_averaging(adjacency_matrix)["laplacian"]
	optimal_lambda2 = second_smallest_eigenvalue(optimal_laplacian)
	print("Optimal lambda2: ", optimal_lambda2)
	print("Optimal lcp_matrix: ", np.identity(n) - optimal_laplacian)
	optimal_convergence_matrix = np.identity(n) - optimal_laplacian - J / n
	print("Optimal convergence_matrix spectrum: ", spectrum(optimal_convergence_matrix))

	laplacian = adjacency_to_laplacian(adjacency_matrix)

	effective_resistance_matrix = compute_effective_resistances(laplacian)
	masked_effective_resistance_matrix = effective_resistance_matrix * adjacency_matrix

	resistance_row_sums = np.sum(masked_effective_resistance_matrix, 1, keepdims=True)

	lcp_matrix = masked_effective_resistance_matrix / resistance_row_sums

	print("Reff lcp_matrix: ", lcp_matrix)
	reff_convergence_matrix = lcp_matrix - J / n
	print("Reff convergence_matrix spectrum: ", spectrum(reff_convergence_matrix))


def diffusion_test(adjacency_matrix, global_iters, iters_per_round):
	num_robots = adjacency_matrix.shape[0]
	final_diffusion_laplacian, final_diffusion_lcp = diffusion_iteration(
		adjacency_matrix, global_iters, iters_per_round)

	global_optimizer_result = fast_linear_averaging(adjacency_matrix=adjacency_matrix)
	optimal_laplacian = global_optimizer_result["laplacian"]

	print("Lambda2 for global optimizer: ", second_smallest_eigenvalue(optimal_laplacian))

	positions = draw_graph_from_adjacency(adjacency_matrix)
	draw_graph_from_laplacian(optimal_laplacian)
	draw_graph_from_laplacian(final_diffusion_laplacian)


def personalized_page_rank_test(adjacency_matrix, alpha=0.1):
	n = adjacency_matrix.shape[0]
	lcp_matrix = naive_lcp_matrix(adjacency_matrix)
	# print("lcp_matrix: ", lcp_matrix)
	seed_vector = build_seed_vector(node_list=[1], dim=n)
	ppr_vec = personalized_page_rank(lcp_matrix, alpha, seed_vector=seed_vector)
	print("ppr_vec: ", ppr_vec)

	ppr_matrix = personalized_page_rank_matrix(lcp_matrix, alpha)
	print("ppr_matrix: ")
	print(ppr_matrix)
	draw_graph_from_adjacency(adjacency_matrix)


def ppr_iteration_test(adjacency_matrix, alpha=0.1, iters=1000, lr=0.5):
	n = adjacency_matrix.shape[0]
	lcp_matrix = naive_lcp_matrix(adjacency_matrix)
	print("lcp_matrix:")
	print(lcp_matrix)
	print("ppr_matrix: ")
	print(personalized_page_rank_matrix(lcp_matrix, alpha))

	new_lcp_matrix = ppr_iteration(lcp_matrix, alpha, iters, lr)
	print("new_lcp_matrix:")
	print(new_lcp_matrix)
	print("row_sums: ", np.sum(new_lcp_matrix, 1))  # TEST
	print("new personalized_page_rank_matrix: ")
	print(personalized_page_rank_matrix(new_lcp_matrix, alpha))

	print("initial lambda2: ", second_smallest_eigenvalue(np.identity(n) - lcp_matrix))
	print("final lambda2: ", second_smallest_eigenvalue(np.identity(n) - new_lcp_matrix))
	print(spectrum(np.identity(n) - new_lcp_matrix))
	results_dict = fast_linear_averaging(adjacency_matrix)
	print("optimal lambda2: ", 1. - results_dict["spectral_gap"])

	optimal_laplacian = results_dict["laplacian"]
	print(spectrum(optimal_laplacian))
	print("optimal ppr_matrix: ")
	print(personalized_page_rank_matrix(np.identity(n) - optimal_laplacian, alpha))

	print("optimal_lcp_matrix: ", np.identity(n) - optimal_laplacian)
	print(optimal_laplacian)

	draw_graph_from_adjacency(adjacency_matrix)
	draw_graph_from_laplacian(np.identity(n) - new_lcp_matrix)
	draw_graph_from_laplacian(optimal_laplacian)


def dist_leader_selection_test(adjacency_matrix, num_leaders, alpha, beta, kappa, mu):
	n = adjacency_matrix.shape[0]
	I = np.identity(n)
	draw_graph_from_adjacency(adjacency_matrix)
	results_dict = distributed_leader_selection(adjacency_matrix, alpha, beta, kappa, mu)

	print(results_dict)

	pagerank_vec = (results_dict["scaled_pagerank_vec"]).tolist()
	rankings_list = (n * results_dict["x"]).tolist()

	original_sorted_rankings = [x[0] for x in sorted(enumerate(pagerank_vec), key=lambda x: x[1], reverse=True)]
	final_sorted_rankings = [x[0] for x in sorted(enumerate(rankings_list), key=lambda x: x[1], reverse=True)]

	print("Original sorted rankings: ", original_sorted_rankings[0:num_leaders])
	print("Final sorted rankings: ", final_sorted_rankings[0:num_leaders])

	orig_augmented_adjacency_matrix, orig_added_edges = connect_nodes(
		adjacency_matrix, original_sorted_rankings[0:num_leaders])
	print("orig_added_edges: ", orig_added_edges)

	new_augmented_adjacency_matrix, new_added_edges = connect_nodes(
		adjacency_matrix, final_sorted_rankings[0:num_leaders])
	print("new_added_edges: ", new_added_edges)

	flda_results_dict = fast_linear_averaging(adjacency_matrix)
	lambda2 = 1. - flda_results_dict["spectral_gap"]
	print("lambda2: ", lambda2)
	lcp_matrix = flda_results_dict["lcp_matrix"]

	orig_augmented_lambda2 = 1. - fast_linear_averaging(orig_augmented_adjacency_matrix)["spectral_gap"]
	print("orig_augmented_lambda2: ", orig_augmented_lambda2)
	#orig_augmented_lcp_matrix = augmented_lcp_matrix(lcp_matrix, original_sorted_rankings[0:num_leaders])
	#print("orig_augmented_lambda2: ", second_smallest_eigenvalue(I - orig_augmented_lcp_matrix))

	new_augmented_lambda2 = 1. - fast_linear_averaging(new_augmented_adjacency_matrix)["spectral_gap"]
	print("new_augmented_lambda2: ", new_augmented_lambda2)
	#new_augmented_lcp_matrix = augmented_lcp_matrix(lcp_matrix, final_sorted_rankings[0:num_leaders])
	#print("new_augmented_lambda2: ", second_smallest_eigenvalue(I - new_augmented_lcp_matrix))

	draw_graph_from_adjacency(adjacency_matrix)

	draw_graph_from_adjacency(new_augmented_adjacency_matrix)


def user_selection_test(adjacency_matrix):

	num_robots = adjacency_matrix.shape[0]
	results_dict = fast_linear_averaging(adjacency_matrix)
	print("lambda2: ", 1. - results_dict["spectral_gap"])

	draw_graph_from_adjacency(adjacency_matrix)
	input_list = input("Select nodes: ").split(',')
	node_list = []
	for node in input_list:
		node_list.append(int(node))
	
	new_adjacency_matrix, new_edges = connect_nodes(adjacency_matrix, node_list)
	new_results_dict = fast_linear_averaging(new_adjacency_matrix)
	print("new lambda2: ", 1. - new_results_dict["spectral_gap"])
	lcp_matrix = new_results_dict["lcp_matrix"]
	for i in range(len(node_list)):
		for j in range(i):
			print("edge: ", node_list[i], node_list[j])
			print("weight: ", lcp_matrix[node_list[i], node_list[j]])
	draw_graph_from_adjacency(adjacency_matrix)
	draw_graph_from_adjacency(new_adjacency_matrix)


def global_page_rank_test(adjacency_matrix, alpha=0.1, beta=0.3):
	n = adjacency_matrix.shape[0]
	results_dict = fast_linear_averaging(adjacency_matrix)
	lcp_matrix = results_dict["lcp_matrix"]
	#print("lcp_matrix: ")
	#print(lcp_matrix)

	rw_matrix = adjacency_to_random_walk(adjacency_matrix)

	adj_ppr_matrix = beta * np.linalg.inv(np.identity(n) - (1.- beta) * rw_matrix)
	print("adj_ppr_matrix: ")
	print(adj_ppr_matrix)

	print("row sums of squares (normalized): ")
	print(np.sum(adj_ppr_matrix * adj_ppr_matrix, 1, keepdims=True) * n)

	lcp_ppr_matrix = beta * np.linalg.inv(np.identity(n) - (1.- beta) * lcp_matrix)
	print("lcp_ppr_matrix: ")
	print(lcp_ppr_matrix)


	adj_pr_vec = global_page_rank(adjacency_matrix, alpha)

	lcp_pr_vec = global_page_rank(lcp_matrix, alpha)
	print("scaled adj_pr_vec: ")
	print(adj_pr_vec * n)

	print("scaled_lcp_pr_vec: ")
	print(lcp_pr_vec * n)
	draw_graph_from_adjacency(adjacency_matrix)


def ppr_sos_test(adjacency_matrix, alpha):
	results_dict = fast_linear_averaging(adjacency_matrix)
	lcp_matrix = results_dict["lcp_matrix"]
	ppr_sos_vector = ppr_sos_vec(lcp_matrix, alpha)
	print("SOS vector: ")
	print(ppr_sos_vector)
	draw_graph_from_adjacency(adjacency_matrix)

if __name__=="__main__":
	# Testing
	np.set_printoptions(precision=3, floatmode="maxprec")
	test_suite = TestSuite()
	num_robots = 100
	
	#adjacency_matrix = test_suite.bicluster(num_robots, 0.8, 0.05)
	#adjacency_matrix = test_suite.cluster_plus_sparse(15, 20, 1., 0.1, 0.9)
	adjacency_matrix = test_suite.robot_proximity_graph(num_robots, min_radius=0.15, max_radius=0.25)
	draw_graph_from_adjacency(adjacency_matrix)
	#global_page_rank_test(adjacency_matrix)
	#user_selection_test(adjacency_matrix)
	
	#adjacency_matrix = test_suite.example_five_node_graph()
	#ppr_sos_test(adjacency_matrix, alpha=0.5)
	#dual_sdp_leader_selection(adjacency_matrix, min_leaders=2)
	#global_page_rank_test(adjacency_matrix, 0.02, 0.3)
	#user_selection_test(adjacency_matrix)


	dist_leader_selection_test(adjacency_matrix, num_leaders=10, alpha=0.5, beta=0.6, kappa=0.8, mu=1.8)

	#adjacency_matrix = test_suite.bicluster(num_robots, 0.8, 0.05)
	#personalized_page_rank_test(adjacency_matrix)


	#41,35,12,3




	#ppr_iteration_test(adjacency_matrix, iters=800)


	#print(adjacency_matrix)
	#adjacency_matrix = test_suite.example_five_node_graph()
	#effective_resistance_weighting(adjacency_matrix=adjacency_matrix, callback=lambda x: np.sqrt(x))
	#personalized_page_rank_test(adjacency_matrix)
	


	# TESTING

	#effective_resistance_weighting(adjacency_matrix)
	#adjacency_matrix = test_suite.example_five_node_graph()


	#asymmetric_reff_weighting(adjacency_matrix)
	#positions = draw_graph_from_adjacency(adjacency_matrix)

	# asym_results_dict = asymmetric_fast_linear_averaging(adjacency_matrix=adjacency_matrix)
	# print("Final asymmetric lambda2: ", 1. - asym_results_dict["gamma"])
	# print("Final asymmetric lcp matrix: ", asym_results_dict["lcp_matrix"])

	#results_dict = fast_linear_averaging(n=num_robots, adjacency_matrix=adjacency_matrix, tol=1e-3)
	#print("Final lambda2: ", 1. - results_dict["spectral_gap"])
	#print("Final lcp matrix: ", np.identity(num_robots) - results_dict["laplacian"])
	#draw_graph_from_adjacency(adjacency_matrix)

	




	#effective_resistance_weighting(adjacency_matrix)
	#diffusion_test(adjacency_matrix, global_iters=200, iters_per_round=6)

	

	
	
	#sensor_locations = test_position_matrix(num_robots=num_robots)
	#plt.scatter(sensor_locations[:, 0], sensor_locations[:, 1])
	#plt.show()
	#cost_matrix = communication_costs(
	#	n=num_robots, sensor_locations=sensor_locations, cost_function=lambda x: x)  # temp - can adjust cost_function

	#test_suite = TestSuite()
	#artificial_cost_matrix = test_suite.example_test_cost_matrix(num_robots)

	# print(cost_matrix)
	# orig_laplacian, reoptimized_laplacian = reoptimizer(
	# 	n=20, cost_matrix=cost_matrix, cost_alpha=3., tol=0.01)
	# print("Original Laplacian: ", orig_laplacian)
	# print("Reoptimized Laplacian: ", reoptimized_laplacian)
	#lambda_versus_alpha_test(num_robots, artificial_cost_matrix, tol=0.0005, alpha_vals=np.arange(0.01, 1.3, 0.02))
	#lambda_versus_alpha_test(num_robots, cost_matrix, tol=0.001, alpha_vals=np.arange(0.1, 0.2, 0.002))
	#lambda_versus_alpha_test(num_robots, cost_matrix, tol=0.001, alpha_vals=np.arange(0.12, 0.8, 0.001))
