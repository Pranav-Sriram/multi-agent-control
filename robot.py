
class Robot:
	def __init__(self, space_dimension, position, velocity, max_sensing_distance, 
		lower_boundary_coordinates, upper_boundary_coordinates, boundary_elasticity):
		self.space_dimension = space_dimension
		self.position = np.array(position)
		self.velocity = velocity
		self.max_sensing_distance = max_sensing_distance
		self.lower_boundary_coordinates = lower_boundary_coordinates
		self.upper_boundary_coordinates = upper_boundary_coordinates
		self.boundary_elasticity = boundary_elasticity


	def move(self, timestep, velocity_delta):
		new_velocity = self.velocity + velocity_delta
		self.position += 0.5 * timestep * (self.velocity + new_velocity)
		self.velocity = new_velocity 
		self.threshold()

	
	def threshold(self):
		for dim in range(self.space_dimension):
			coord = self.position[dim]
			if coord =< lower_boundary_coordinates[dim]:
				self.position[dim] = lower_boundary_coordinates[dim]
				self.velocity *= -self.boundary_elasticity
			if coord >= upper_boundary_coordinates[dim]:
				self.position[dim] = upper_boundary_coordinates[dim]
				self.velocity[dim] *= -self.boundary_elasticity


	def damp_motion(self, damping_factor):
		self.velocity *= damping_factor






