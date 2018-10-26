
class MotionCalculator:
	def __init__():
		pass

	def position_to_squared_distance(robot_positions):
		""" Returns a matrix of squared pairwise distances 
		between robots given their positions."""
		pass

	def compute_all_net_forces(self):
		particle_repulsive_forces = self.compute_all_particle_repulsive_forces()
		wall_forces = self.compute_all_wall_forces()
		# tension_forces = self.compute_all_pairwise_tensions()
		return particle_repulsive_forces + wall_forces # + tension_forces


	def acceleration(self, force):
		return force / self.mass_constant # may change to variable masses later


	# PArt 2: Force computation details
	def compute_all_particle_repulsive_forces(self):
		particle_forces = np.zeros((self.num_points, self.space_dimension))
		for i in range(self.num_points):
			particle_forces[i, :] = self.compute_net_repulsive_force(i)
		return particle_forces


	def compute_all_wall_forces(self):
		wall_forces = np.zeros((self.num_points, self.space_dimension))
		for i in range(self.num_points):
			wall_forces[i, :] = self.compute_wall_force(self.points[i, :])
		return wall_forces


	def compute_pairwise_tensions(self, forces):

		for i in range(1, self.num_points):
			point_i = self.points[i, :]
			for j in range(i):
				active, max_distance = self.active_distance_constraints[(i, j)]
				if active:
					point_j = self.points[j, :]
					dist_ij = compute_distance(point_i, point_j)
					if dist_ij > max_distance:
						






	def compute_net_repulsive_force(self, i):
		force_vector = np.zeros(self.space_dimension)
		assert(0 <= i and i < self.num_points)
		for j in range(0, self.num_points):
			if j != i:
				force_vector += self.compute_pairwise_force(
					self.points[i, :], self.points[j, :], (i, j))  # force exerted by point j on point i
		return force_vector

	def compute_wall_force(self, point_i):
		"""Computes repulsive forces from walls of bounding polygon."""
		wall_force = np.zeros(self.space_dimension)
		for dim in range(self.space_dimension):
			for direction in [-1, 1]:
				projection = np.copy(point_i)
				projection[dim] = direction
				wall_force += self.compute_pairwise_force(point_i, projection)
		return wall_force


	def compute_pairwise_force(self, point_i, point_j, update_energy=None):
		"""Computes force exerted by point_j on point_i.
		Args:
			point_i: vector of length space_dimension
			point_j: vector of length space_dimension
			update_energy: if not None, it's a pair of indices (i, j) denoting
			the indices of the points in the self.points matrix.
		"""

		diff_vector = point_i - point_j  # repulsion so i is head of vector
		distance = (np.sum(diff_vector * diff_vector)) ** 0.5  # Euclidean dist
		if update_energy:
			i, j = update_energy
			self.energies[i, j] = 1./(distance + 1e-7)  # Update mutual potential energy. Also prevent denominator from becoming 0
		return ((self.mass_constant)**2) * diff_vector/(distance ** 3 + 0.0001)  # smoother repulsion 
			
