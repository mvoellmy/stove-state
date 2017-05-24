import numpy as np
from random import randint, random
import math


class ParticleFilter():
    def __init__(self, nr_of_particles=200):
        self.nr_of_particles = nr_of_particles
        self.particles = np.random.randint(3, size=self.nr_of_particles)  # 0 = empty, 1 = pan, 2 = lid
        self.new_particles = np.random.randint(3, size=self.nr_of_particles)  # 0 = empty, 1 = pan, 2 = lid
        self.weights = np.ones(self.nr_of_particles)
        self.min_nr_particles = 2
        self.prob_same_state = 0.8
        self.prob_different_state = (1-0.8)/2
        self.particles_distribution = [np.count_nonzero(self.particles == 0),
                                       np.count_nonzero(self.particles == 1),
                                       np.count_nonzero(self.particles == 2)]

    def update_particles(self, gesture):
        if 'place' in gesture:
            g = 1
        elif 'remove' in gesture:
            g = -1
        else:
            g = 0

        self.particles[:] = [max(min(particle + g*randint(0, 2), 2), 0) for particle in self.particles]

    def update_weights(self, pan_label_name):
        if 'plate' in pan_label_name:
            z = 0
        elif 'pan' in pan_label_name:
            z = 1
        elif 'lid' in pan_label_name:
            z = 2

        for i, particle in enumerate(self.particles):
            if particle == z:
                self.weights[i] = self.prob_same_state
            else:
                self.weights[i] = self.prob_different_state

        # Normalize weights
        self.weights = self.weights / np.linalg.norm(self.weights)

        if math.ceil(100*np.linalg.norm(self.weights)/100) != 1:
            print('ERROR: normalization of weights failed {}'.format(np.linalg.norm(self.weights)))

        self.resample()
        self.roughen()
        # self.plot_particles()

    def resample(self):

        for i, particle in enumerate(self.particles):
            r = random()
            weight_sum = 0
            for j, weight in enumerate(self.weights):
                weight_sum += weight
                if weight_sum > r:
                    self.new_particles[i] = self.particles[j]
                    break

        self.weights = np.ones(self.nr_of_particles)
        self.particles = np.copy(self.new_particles)

    def roughen(self):
        self.count_particles()
        for particle_state, count in enumerate(self.particles_distribution):
            if count == 0:
                for i in range(0, self.min_nr_particles):
                    self.particles[randint(0, self.nr_of_particles-1)] = particle_state

    def plot_particles(self):
        self.count_particles()
        # Compare particles to label nrs and count occurrences where they are present.
        print('E: {}\tP: {}\tL: {}'.format(self.particles_distribution[0],
                                           self.particles_distribution[1],
                                           self.particles_distribution[2]))

    def count_particles(self):
        self.particles_distribution = [np.count_nonzero(self.particles == 0),
                                       np.count_nonzero(self.particles == 1),
                                       np.count_nonzero(self.particles == 2)]

        return self.particles_distribution
