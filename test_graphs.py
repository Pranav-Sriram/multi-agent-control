import numpy as np
import networkx as nx

class TestSuite:

	def __init__(self):
		return 

	def v_graph(self):
		return np.array([
			[0., 1., 0.],
			[1., 0., 1.],
			[0., 1., 0.]
			])

	def example_five_node_graph(self):
		return np.array([[0., 1., 1., 1., 0.],
			[1., 0., 1., 0., 0.],
			[1., 1., 0., 1., 0.],
			[1., 0., 1., 0., 1.],
			[0., 0., 0., 1., 0.]])

	def clique_plus_one(self, clique_size, connector=0):
		n = clique_size
		adjacency_matrix = np.ones((n+1, n+1))
		for i in range(n+1):
			adjacency_matrix[i, i] = 0
		for i in range(n):
			adjacency_matrix[i, n] = 0
			adjacency_matrix[n, i] = 0
		adjacency_matrix[connector, n] = 1.
		adjacency_matrix[n, connector] = 1.
		return adjacency_matrix


	def make_random_adjacency_matrix(self, num_robots, prob):
		""" Makes a random adjacency matrix.
		Each edge is present with probability prob."""
		adjacency_matrix = np.zeros((num_robots, num_robots))
		for i in range(1, num_robots):
			for j in range(i):
				if np.random.random() < prob:
					adjacency_matrix[i, j] = 1.
					adjacency_matrix[j, i] = 1.
		return adjacency_matrix


	def example_test_cost_matrix(self, num_robots):
		cost_matrix = np.random.random((num_robots, num_robots))
		for i in range(num_robots):
			for j in range(num_robots):
				if np.random.random() < 0.03:
					cost_matrix[i, j] = 8.
					cost_matrix[j, i] = 8.
		return cost_matrix


	def bicluster(self, num_robots, p=0.7, q=0.05):
		adjacency_matrix = np.zeros((num_robots, num_robots))
		for i in range(1, num_robots):
			for j in range(i):
				if (i - (num_robots-1) / 2.) * (j - (num_robots-1) / 2.) <= 0:
					adjacency_matrix[i, j] = 1. if np.random.random() < q else 0.
				else:
					adjacency_matrix[i, j] = 1. if np.random.random() < p else 0.
				adjacency_matrix[j, i] = adjacency_matrix[i, j]
		return adjacency_matrix


	def dumbbell(self, num_robots):
		adjacency_matrix = np.zeros((num_robots, num_robots))
		for i in range(1, num_robots):
			for j in range(i):
				if i < num_robots / 2 or j >= num_robots / 2:
					adjacency_matrix[i, j] = 1.
					adjacency_matrix[j, i] = 1.
		adjacency_matrix[0, num_robots-1] = 1.
		adjacency_matrix[num_robots-1, 0] = 1.
		return adjacency_matrix


	def cluster_plus_sparse(self, cluster_size, sparse_size, cc_prob, sc_prob, ss_prob):
		n = cluster_size + sparse_size
		adjacency_matrix = np.zeros((n, n))
		for i in range(1, n):
			for j in range(i):
				if i < cluster_size:  # cluster-cluster
					prob = cc_prob 
				else:
					prob = sc_prob if j < cluster_size else ss_prob

				adjacency_matrix[i, j] = 1. if np.random.random() < prob else 0.
				adjacency_matrix[j, i] = adjacency_matrix[i, j]
		return adjacency_matrix


	def robot_proximity_graph(self, num_robots, min_radius=0.1, max_radius=0.5):
		sensing_radii = np.random.uniform(low=min_radius, high=max_radius, size=(num_robots))
		positions = np.random.random(size=(num_robots, 2))
		adjacency_matrix = np.zeros((num_robots, num_robots))
		for i in range(1, num_robots):
			for j in range(i):
				dist_ij = np.sum((positions[i, :] - positions[j, :]) ** 2) ** 0.5
				if dist_ij < sensing_radii[i] and dist_ij < sensing_radii[j]:
					adjacency_matrix[i, j] = 1.
					adjacency_matrix[j, i] = 1.
		return adjacency_matrix


						