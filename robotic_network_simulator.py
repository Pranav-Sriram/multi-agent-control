import numpy as np
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import robot


class RoboticNetworkSimulator:
	def __init__(self, num_robots, space_dimension, robot_positions=None, 
		velocities=None, max_robot_sensing_distance=float("inf"), axis1=0, 
		axis2=1, view="standard", verbose=False, 
		lower_boundary_coordinates, upper_boundary_coordinates, boundary_elasticity):
		self.num_robots = num_robots
		self.space_dimension = space_dimension
		self.energies = np.zeros((num_points, num_points))  # "Normalized" energies, i.e. assuming unit mass 
		self.axis1 = axis1
		self.axis2 = axis2 
		self.verbose = verbose

		self.active_distance_constraints = {(i, j) : (False, -1) for i in range(
			num_points) for j in range(num_points)}  # map between pairs of points to pairs (bool, value)
		# indicating whether there is an active distance constraint between them

		if robot_positions is None:  
			robot_positions = np.random.uniform(-1.0, 1.0, (num_robots, space_dimension))  # temporary 
		self.robot_positions = robot_positions
			
		if velocities is None:
			velocities = np.zeros((num_points, space_dimension))
		self.velocities = velocities
		
		self.robots = []
		for i in range(num_robots):
			self.robots.append(
				Robot(
					space_dimension=space_dimension,
					position=robot_positions[i]
					velocity=velocities[i],
					max_sensing_distance=max_robot_sensing_distance,
					lower_boundary_coordinates=lower_boundary_coordinates,
					upper_boundary_coordinates=upper_boundary_coordinates,
					boundary_elasticity=boundary_elasticity
				)
			)

		self.view = view
		if self.verbose:
			print("Initial positions: ")
			print(robot_positions)

		
	# Set the display and physical constants
	def set_physical_parameters(self, damping_factor=0.9, damping_freq=5, num_iters=3000, frame_rate=1, 
		interval=10, pause_duration=3.0, time_step=5e-5, mass_constant=0.5, speedup_factor=1.0):
		self.damping_factor = damping_factor
		self.damping_freq = damping_freq
		self.num_iters = num_iters
		self.frame_rate = frame_rate  # number of iterations of physical updates per animation frame
		self.interval = interval  # interval between successive frames (in milliseconds)
		self.pause_duration = pause_duration
		self.time_step = time_step
		self.mass_constant = mass_constant
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

	#
	def update_physical_parameters(self):
		"""Computes net forces on each point and updates motion parameters accordingly."""
		net_forces = self.compute_all_net_forces()
		for i in range(0, self.num_points):
			force = net_forces[i, :]
			velocity_delta = self.acceleration(i, force) * self.time_step  # force exerted changes velocity. Old val erased each time
			self.robots[i].move(self.time_step, velocity_delta)


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
			self.update_physical_parameters()  

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


# PART 7: MAIN
if __name__=="__main__":
	model1 = RoboticNetworkSimulator(num_points=12, space_dimension=2, view="standard")
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
