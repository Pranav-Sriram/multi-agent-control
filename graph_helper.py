# graph helper functions
import numpy as np
from cvxpy import *
import networkx as nx
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.sparse import linalg as linalg



# Helper functions

# Basic Conversion and Creation
def test_position_matrix(num_robots):
	return 5. * np.random.random(size=(num_robots, 2))


def threshold_weights(matrix, tol=1e-4):
	"""Zeroes out entries of matrix with absolute value less than tol.""" 
	return np.where(np.abs(matrix) > tol, matrix, np.zeros_like(matrix))


def laplacian_to_adjacency(laplacian):
	adj = -laplacian
	for i in range(laplacian.shape[0]):
		adj[i, i] = 0.  # zero out diagonals
	return adj


def adjacency_to_laplacian(adjacency_matrix):
	n = adjacency_matrix.shape[0]
	laplacian = -adjacency_matrix
	for i in range(n):
		laplacian[i, i] = np.sum(adjacency_matrix[i, :])
	return laplacian


def adjacency_to_random_walk(adjacency_matrix):
	return adjacency_matrix / np.sum(adjacency_matrix, 0, keepdims=True)


def laplacian_to_unweighted_adjacency(laplacian, tol):
	return np.where(np.abs(laplacian) > tol, np.ones_like(laplacian), np.zeros_like(laplacian))


def naive_averaging_laplacian(adjacency_matrix):
	n = np.shape(adjacency_matrix)[0]
	degrees = np.sum(adjacency_matrix, 1, keepdims=True)
	max_degree = np.amax(degrees)

	scaled_matrix = adjacency_matrix / (max_degree + 1.)
	return adjacency_to_laplacian(scaled_matrix)


def naive_lcp_matrix(adjacency_matrix):
	return np.identity(adjacency_matrix.shape[0]) - naive_averaging_laplacian(adjacency_matrix)


def connect_nodes(adjacency_matrix, node_list):
	n = adjacency_matrix.shape[0]
	new_matrix = np.zeros((n, n))
	new_matrix[0:n, 0:n] = np.copy(adjacency_matrix)
	new_edges = 0

	for i in range(1, len(node_list)):
		for j in range(i):
			if new_matrix[node_list[i], node_list[j]] != 1.:
				new_edges += 1
				new_matrix[node_list[i], node_list[j]] = 1.
				new_matrix[node_list[j], node_list[i]] = 1.
	return new_matrix, new_edges 


def pool_neighbors(adjacency_matrix, node_list):
	n = adjacency_matrix.shape[0]
	added_edges = 0
	new_matrix = np.copy(adjacency_matrix)
	for i, node_i in enumerate(node_list):
		for j, node_j in enumerate(node_list):
			if j < i:
				for k in range(n):
					if adjacency_matrix[i, k] == 1. and adjacency_matrix[j, k] != 1.:
						new_matrix[j, k] = 1.
						new_matrix[k, j] = 1.
						added_edges += 1
					elif adjacency_matrix[j, k] == 1. and adjacency_matrix[i, k] != 1.:
						new_matrix[i, k] = 1.
						new_matrix[k, i] = 1.
						added_edges += 1
	return new_matrix, added_edges



def augmented_lcp_matrix(lcp_matrix, node_list):
	n = lcp_matrix.shape[0]
	m = len(node_list)
	Q = np.zeros((n, n))

	for i in range(n):
		if i not in node_list:
			Q[i, i] = 1.

	for i, node_i in enumerate(node_list):
		for j, node_j in enumerate(node_list):
			if j < i:
				Q[i, j] = 1./m
				Q[j, i] = 1./m
	return np.matmul(Q, lcp_matrix) 





def adjacency_to_networkx(adjacency_matrix):
	"""Converts numpy array to a networkx graph object."""
	n = adjacency_matrix.shape[0]  # number of nodes
	graph = nx.Graph()
	for i in range(n):
		graph.add_node(i)
	for i in range(1, n):
		for j in range(i):
			if adjacency_matrix[i, j] > 0:
				graph.add_edge(i, j, weight=adjacency_matrix[i, j])
	return graph


def laplacian_to_networkx(laplacian):
	return adjacency_to_networkx(laplacian_to_adjacency(laplacian))


# Drawing

def draw_graph_from_adjacency(adjacency_matrix, positions=None):
	graph = adjacency_to_networkx(adjacency_matrix)
	pos = positions if positions is not None else nx.spring_layout(graph)
	nx.draw_networkx(graph, pos=pos, edge_cmap=plt.get_cmap("jet"), with_labels=True)
	plt.show()
	return pos


def draw_graph_from_laplacian(laplacian, positions=None):
	return draw_graph_from_adjacency(laplacian_to_adjacency(laplacian), positions=positions)


# Spectral Computations
def laplacian_pseudoinverse(laplacian):
	n = np.shape(laplacian)[0]
	J = np.ones((n, n))
	return np.linalg.inv(laplacian + J/n) - J/n


def compute_effective_resistances(laplacian):
	"""Returns matrix of pairwise effective resistances of graph given the Laplacian."""
	n = np.shape(laplacian)[0]
	L_pseudoinv = laplacian_pseudoinverse(laplacian)
	diag = np.diagonal(L_pseudoinv).copy()
	return diag.reshape((n, 1)) + diag.reshape((1, n)) - 2 * L_pseudoinv  # broadcasting ftw ;)


def smallest_eigenvalue(matrix):
	"""Note: matrix must be PSD"""
	eigval, eigvec = linalg.eigsh(matrix, k=1, which="SM", maxiter=4000, tol=1e-3)
	return eigval[0]


def second_smallest_eigenvalue(laplacian):
	eigvals, eigvecs = linalg.eigs(laplacian, k=2, which="SM")  # smallest magnitude ordering
	return eigvals[1]


def largest_eigenvalue(laplacian):
	eigval, eigvec = linalg.eigs(laplacian, k=1, which="LM")  # largest magnitude ordering
	return eigval[0]


def spectrum(matrix):
	"Returns all eigenvalues of a given matrix."
	eigvals, eigvecs = np.linalg.eig(matrix)
	return eigvals


def diffusion_process_simulation(diffusion_matrix, initial_node_vals, iters):
	"""Simulates a diffusion/random walk for iters steps."""
	new_node_vals = initial_node_vals
	for i in range(iters):
		new_node_vals = diffusion_matrix.dot(new_node_vals)
	return new_node_vals


#def distributed_lcp_simulation(lcp_matrix, iters):
	#initial_node_vals = np.identity()

def build_seed_vector(node_list, dim):
	"""Creates an indicator vector with 1s in the positions given in node_list"""
	vec = np.zeros(dim)
	if node_list is None or len(node_list) == 0: return vec
	for node in node_list:
		vec[node] = 1.
	return vec / len(node_list)


def global_page_rank(diffusion_matrix, alpha):
	n = diffusion_matrix.shape[0]
	new_diffusion_matrix = np.copy(diffusion_matrix)
	for i in range(n):
		new_diffusion_matrix[i, i] = 0.

	new_diffusion_matrix /= np.sum(new_diffusion_matrix, 0, keepdims=True)  # normalize
	return personalized_page_rank(new_diffusion_matrix, alpha, np.ones(n)/n)


def personalized_page_rank(diffusion_matrix, alpha, seed_vector):
	n = diffusion_matrix.shape[0]
	kernel = np.identity(n) - (1. - alpha) * diffusion_matrix
	return alpha * np.dot(np.linalg.inv(kernel), seed_vector)


def personalized_page_rank_matrix(diffusion_matrix, alpha):
	n = diffusion_matrix.shape[0]
	return alpha * np.linalg.inv(np.identity(n) - (1. - alpha) * diffusion_matrix)


def ppr_sos_vec(lcp_matrix, alpha):
	n = lcp_matrix.shape[0]
	lcp_ppr_matrix = alpha * np.linalg.inv(np.identity(n) - (1.- alpha) * lcp_matrix)
	row_sos = np.sum(lcp_ppr_matrix * lcp_ppr_matrix, 1, keepdims=True)
	return n * row_sos / np.sum(row_sos)

def ppr_double_squared(lcp_matrix, alpha):
	sos_vec = ppr_sos_vec(lcp_matrix, alpha)
	return sos_vec * sos_vec / np.sum(sos_vec * sos_vec)


