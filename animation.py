import numpy as np
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

class Animator:
	def __init__(xlim, ylim):
		self.xlim = xlim
		self.lim = ylim

	def plot_points(self, points):
		plt.plot(points[:, 0], points[:, 1], 'bo')
		plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])  # todo
		plt.show()

	def dynamics(self):
		fig = plt.figure()
		plt.xlim(self.xlim)
		plt.ylim(self.ylim)
		points_ani = animation.FuncAnimation(fig, func=self.animation_callback, frames=self.num_frames, 
			fargs=None, interval=self.interval, repeat=False)
		plt.show()

