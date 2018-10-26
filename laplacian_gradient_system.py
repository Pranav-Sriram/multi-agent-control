import numpy as np
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

class DynamicRepulsionModel:
	def __init__(self, num_points, space_dimension, points=None, velocities=None, axis1=0, axis2=1, view="standard", verbose=False):
		self.num_points = num_points
		self.space_dimension = space_dimension
		
		self.energies = np.zeros((num_points, num_points))  # "Normalized" energies, i.e. mass independent. 
		self.velocity_deltas = np.zeros((num_points, space_dimension))
		self.axis1 = axis1
		self.axis2 = axis2 
		self.verbose = verbose

		self.active_distance_constraints = {(i, j) : (False, -1) for i in range(
			num_points) for j in range(num_points)}  # map between pairs of points to pairs (bool, value)
		# indicating whether there is an active distance constraint between them

		if points is None:  # each row is a point
			self.points = np.random.uniform(-1.0, 1.0, (num_points, space_dimension))
			self.initial_points = np.copy(self.points)  # Deep copy. Preserve this
		else:
			self.points = points   # TODO - handle bad input by squashing to [-1, 1] appropriately
			self.initial_points = np.copy(self.points)

		if velocities is None:
			self.velocities = np.zeros((num_points, space_dimension))
		else:
			self.velocities = velocities

		self.view = view
		if(self.verbose):
			print("Initial points: ")
			print(self.points)

	# Set the display and physical constants
	def set_physical_parameters(self, damping_factor=0.9, damping_freq=5, num_iters=3000, frame_rate=1, 
		interval=10, pause_duration=3.0, time_step=5e-5, mass_constant=0.5, elasticity=0.0, speedup_factor=1.0):
		self.damping_factor = damping_factor
		self.damping_freq = damping_freq
		self.num_iters = num_iters
		self.frame_rate = frame_rate  # number of iterations of physical updates per animation frame
		self.interval = interval  # interval between successive frames (in milliseconds)
		self.pause_duration = pause_duration
		self.time_step = time_step
		self.mass_constant = mass_constant
		self.elasticity = elasticity
		self.speedup_factor = speedup_factor
		self.num_frames = int(num_iters / frame_rate)

    # Set paramaters controlling frequency of collecting and printing information
	def set_information_parameters(self, energy_compute_freq=25, print_freq=100, verbose=False, show_plots=True):
		self.print_freq = print_freq
		self.energy_compute_freq = energy_compute_freq
		self.num_energy_intervals = int(self.num_frames / energy_compute_freq) + 1
		self.total_energies = np.zeros(self.num_energy_intervals)
		self.time_axis = np.arange(0, self.num_energy_intervals)
		self.show_plots = show_plots

	def set_active_distance_constraints(active_distance_constraints):
		self.active_distance_constraints = active_distance_constraints

	# PART 1: MOTION - high level implementation

	def update_physical_parameters(self):
		"""Computes net forces on each point and updates motion parameters accordingly."""
		net_forces = self.compute_all_net_forces()
		for i in range(0, self.num_points):
			force = net_forces[i, :]
			self.velocity_deltas[i, :] = self.acceleration(i, force) * self.time_step # force exerted changes velocity. Old val erased each time
		self.move_points(self.time_step)  # all points take step in direction of velocity

	def compute_all_net_forces(self):
		particle_repulsive_forces = self.compute_all_particle_repulsive_forces()
		wall_forces = self.compute_all_wall_forces()
		# tension_forces = self.compute_all_pairwise_tensions()
		return particle_repulsive_forces + wall_forces  # todo: add tension_forces

	def acceleration(self, i, force):
		return force / self.mass_constant # may change to variable masses later

	# Updates points, velocities using current velocities and velocity deltas
	def move_points(self, time_step_param):
		self.points += time_step_param * (self.velocities + 0.5 * self.velocity_deltas)  # s = ut + 0.5at^2
		self.velocities += self.velocity_deltas  # v = u + at
		for i in range(0, self.num_points):  
			for j in range(0, self.space_dimension):
				self.points[i, j] = self.threshold(i, j)  # threshold both points and velocities
		
	def threshold(self, i, j):		
		x = self.points[i, j]
		if (x < -1):
			self.velocities[i, j] *= -self.elasticity  # velocity in this direction reverses and possibly dissipates upon collision with wall
			return -1
		if(x > 1):
			self.velocities[i, j] *= -self.elasticity
			return 1
		return x

	def damp_motion(self):
		self.velocities *= self.damping_factor
		self.velocity_deltas *= self.damping_factor

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
			
	
	# PART 3: VANILLA DYNAMICS (No plotting, just final info)
	def vanilla_dynamics(self, polarize=False):
		for iter in range(0, self.num_iters):
			self.update_physical_parameters()
			# damp motion
			if(iter % (self.damping_freq * self.frame_rate) == 0):  # Damp motion every iter frames
				self.damp_motion() 
			# log energies
			if (iter % (self.energy_compute_freq * self.frame_rate) == 0):
				self.total_energies[iter/self.energy_compute_freq] = np.sum(self.energies/2)
		
		if(polarize):
			self.polarize_all()
		return self.points

	# PART 4: AXES AND PROJECTIONS
	def principal_projection(self):
		"""Perform SVD (PCA) to obtain a visualization of high dimensional points in 2D"""
		matrix = self.initial_points if self.view == "fixed_pca" else self.points
		# matrix = self.initial_points  # CHANGE THIS
		U, S, V = np.linalg.svd(matrix, full_matrices=0)  # (N x S; S x S; S x S)
		s_indices = np.argsort(S)
		index1 = s_indices[s_indices.size-1]
		index2 = s_indices[s_indices.size-2]

		# Now working with points (svd may be with initial_points)
		point_lengths = np.linalg.norm(self.points, axis=1)  # N norms
		projection_axis1 =  (self.points).dot(V[index1, :]) / (self.space_dimension**0.5) # transpose done automatically in numpy
		projection_axis2 = (self.points).dot(V[index2, :]) / (self.space_dimension**0.5)
		return (projection_axis1, projection_axis2)

	def get_axes(self):
		if self.view=="principal_components" or self.view=="both":
			first_comp, second_comp = self.principal_projection()
			if(self.view=="both"):
				first_dim = self.points[:, self.axis1] 
				second_dim = self.points[:, self.axis2]
				return (first_dim, second_dim, first_comp, second_comp)
			else:
				return (first_comp, second_comp)
		else:
			first_dim = self.points[:, self.axis1] 
			second_dim = self.points[:, self.axis2]
			return first_dim, second_dim

		# elif(self.view=="fixed_pca"):
		# 	first_comp, second_comp = self.principal_projection()
		# 	return first_comp, second_comp

	
	# PART 5: PLOTTING AND ANIMATED DYNAMICS
	def plot_points(self):
		if(self.space_dimension > 1):
			if(self.view == "both"):
				first_dim, second_dim, first_comp, second_comp = self.get_axes()
				plt.plot(first_dim, second_dim, 'ro', first_comp, second_comp, 'bo')
			else:
				first_dim, second_dim = self.get_axes()
				plt.plot(first_dim, second_dim, 'bo')
			
			plt.axis([-1., 1., -1., 1.])
			plt.show()

	# Run the dynamics model
	def dynamics(self):
		fig = plt.figure()
		plt.xlim(-1., 1.)
		plt.ylim(-1., 1.)
		points_ani = animation.FuncAnimation(fig, func=self.animation_callback, frames=self.num_frames, 
			fargs=None, interval=self.interval, repeat=False)
		plt.show()

	
	def animation_callback(self, num):
		""" Callback registered with the FuncAnimation.

		Implements the backend logic for updating points between frames."""

		# pause at start 
		if num == 1:
			plt.pause(self.pause_duration)

		# do some setup - clear previous screen but restore axis limits
		plt.clf()
		plt.xlim(-1., 1.)
		plt.ylim(-1., 1.)
		
		annealed_time_step = self.time_step * (1 + (self.speedup_factor - 1) * num/self.num_frames) 

		# Loop for num_frame frames, updating physical information
		for iter in range(0, self.frame_rate):
			self.update_physical_parameters()  # used to pass in annealed_time_step; now don't use annealing at all

		# Damp motion
		if num % self.damping_freq == 0:
			self.damp_motion()

		# Handle energy logging 
		if num % self.energy_compute_freq == 0:
			print(num)
			self.total_energies[int(num/self.energy_compute_freq)] = np.sum(self.energies)/2

		#Get axes
		if self.view == "both":
			first_dim, second_dim, first_comp, second_comp = self.get_axes()
		else:
			first_dim, second_dim = self.get_axes()

		# Return a plot
		if self.view=="both":
			return plt.plot(first_dim, second_dim, 'bo', first_comp, second_comp, 'ro')
		else:
			return plt.plot(first_dim, second_dim, 'go')

			
	# PART 6: FINAL SUMMARIES

	def polarize_all(self):
		""" Maps final positions to corners of the hypercube.
		This function is used in applications where the goal of the simulation
		is to perform a gradient-based optimization over continuous variables,
		but whose output must be integer-valued (which was in fact the original purpose of this
		project!). """

		for i in range(0, self.num_points):
			for j in range(0, self.space_dimension):
				self.points[i, j] = 1. if self.points[i, j] >= 0 else -1.
				
	def show_final_summaries(self, show_points=True, show_vel=True, show_coord=True, show_energies=True):
		if show_points:
			print("Final Points: \n")
			print(self.points) 
		
		if show_vel:
			print("Final Velocities: \n")
			print(self.velocities)
		
		if show_coord:
			print("Final coordinates: \n")
			if(self.view=="both"):
				first_dim, second_dim, first_comp, second_comp = self.get_axes()
				print(first_dim)
				print(second_dim)
				print(first_comp)
				print(second_comp)
			else:
				first_dim, second_dim = self.get_axes()
				print(first_dim)
				print(second_dim)

		if show_energies:
			print("Energies: ")
			print(self.total_energies)
		plt.plot(self.time_axis, self.total_energies, 'b--')
		max_energy = np.amax(self.total_energies)
		plt.axis([0, self.num_energy_intervals, 0, max_energy*1.1])
		plt.show()


# PART 6: MAIN
if __name__=="__main__":
	model1 = Dynamic_Repulsion_Model(num_points=12, space_dimension=2, view="standard")
	model1.set_physical_parameters(damping_factor=0.9, damping_freq=5, num_iters=1000, frame_rate=1, mass_constant=2e4, 
		time_step=5e-5, interval=5, elasticity=0.88)
	model1.set_information_parameters()
	model1.dynamics()
	model1.show_final_summaries()
	#final_points = model1.vanilla_dynamics(polarize=True)
	#print final_points 
	# model1.show_final_summaries(show_vel=False, show_coord=False)


	# Constants that work fairly well:
	# model1 = Dynamic_Repulsion_Model(10, 50, view="standard")
	# model1.set_physical_parameters(damping_factor=0.97, damping_freq=20, frame_rate=1, mass_constant=5e5, time_step=1e-4, interval=5)

