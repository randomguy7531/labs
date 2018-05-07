from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np
import scipy.stats
import math

def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments: 
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    robot_delta_x = odom[0]
    robot_delta_y = odom[1]
    robot_delta_h = odom[2]

    motion_particles = []
    # move every particles world coordinates in correspondance with the robot's
    for particle in particles:
        particle_delta_x = robot_delta_x * math.cos(math.radians(particle.h)) - \
                           robot_delta_y * math.sin(math.radians(particle.h))
        particle_delta_y = robot_delta_y * math.cos(math.radians(particle.h)) + \
                           robot_delta_x * math.sin(math.radians(particle.h))
        particle_delta_h = robot_delta_h
        motion_particles.append(Particle(particle.x + particle_delta_x,
                                         particle.y + particle_delta_y,
                                         particle.h + particle_delta_h))
    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information, 
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    # note: algo derived from the question asked on canvas
    # https://canvas.uw.edu/courses/1199396/discussion_topics/4316627

    num_particles = len(particles)
    particle_conditional_probabilities = []
    num_measured_markers = len(measured_marker_list)
    # compute the probability of each particle being the robot, given the measurement value
    for particle in particles:
        particle_read_markers = particle.read_markers(grid)
        num_particle_read_markers = len(particle_read_markers)
        # if a particle is outside of the grid, then it obviously has a probability of zero
        if isParticleOutsideGrid(particle, grid):
            particle_conditional_probabilities.append(0)
        # if we see no marker, but the particle sees a marker, its probability is dropped to zero
        elif num_measured_markers == 0 and num_particle_read_markers > 0:
            particle_conditional_probabilities.append(0)
        # if we see a marker, but the particle sees no markers, its probability is dropped to zero
        elif num_measured_markers > 0 and num_particle_read_markers == 0:
            particle_conditional_probabilities.append(0)
        # if we did not see a marker, all particles that did not see markers are equally likely
        elif num_measured_markers == 0 and num_particle_read_markers == 0:
            particle_conditional_probabilities.append(1)
        # if we both get a marker measurement, and the particle sees at least one marker, we have math to do
        else:
            joint_markers_measurement_probability = 1
            for measured_marker in measured_marker_list:
                # we don't know what marker we measured, so compute the probability as a sum of gaussians
                cur_sum = 0
                for grid_marker in particle_read_markers:
                    cur_sum += getConditionalProbOfMarker(measured_marker, grid_marker, particle)
                # update the joint probability with our measurement
                joint_markers_measurement_probability *= cur_sum
            particle_conditional_probabilities.append(joint_markers_measurement_probability)

    # resample from the updated distribution of particles
    normalized_particle_probabilities = [float(particle_prob) / sum(particle_conditional_probabilities)
                                         for particle_prob in particle_conditional_probabilities]
    num_salt_particles = int(math.floor(num_particles * 0.05))
    resampled_particles = np.random.choice(particles,
                                           num_particles - num_salt_particles,
                                           p=normalized_particle_probabilities)

    trans_perturbation_sigma = 0.1  # translational err in inch (grid unit)
    rot_perturbation_sigma = 5  # rotational err in deg

    # add some noise to the resampled particles
    # otherwise, we will simply coalesce to a small number of repeat hypotheses
    resampled_particles = [perturbParticle(resampled_particle, trans_perturbation_sigma, rot_perturbation_sigma)
                           for resampled_particle in resampled_particles]

    # salt some new particles with random values
    # avoids getting caught in wrong answers when the initial guesses are all wrong
    salt_particles = Particle.create_random(num_salt_particles, grid)

    return np.concatenate([resampled_particles, salt_particles])

def getConditionalProbOfMarker(robot_measured_marker, grid_marker, particle):
    # note: algo derived from the question asked on canvas
    # https://canvas.uw.edu/courses/1199396/discussion_topics/4316627

    # marker measurement Gaussian noise model (from settings py)
    MARKER_TRANS_SIGMA = 0.5  # translational err in inch (grid unit)
    MARKER_ROT_SIGMA = 5  # rotational err in deg

    # get the conditional probabilities of each measurement from the Gaussian PDF
    x_probability = scipy.stats.norm.pdf(robot_measured_marker[0], grid_marker[0], MARKER_TRANS_SIGMA)
    y_probability = scipy.stats.norm.pdf(robot_measured_marker[1], grid_marker[1], MARKER_TRANS_SIGMA)
    h_probability = scipy.stats.norm.pdf(robot_measured_marker[2], grid_marker[2], MARKER_ROT_SIGMA)

    # return the join probabilities of the measurements
    return x_probability * y_probability * h_probability

def isParticleOutsideGrid(particle, grid):
    outside_x_bounds = particle.x > grid.width or particle.x < 0
    outside_y_bounds = particle.y > grid.height or particle.y < 0
    return outside_x_bounds or outside_y_bounds

def perturbParticle(particle, translate_noise, heading_noise):
    new_x = add_gaussian_noise(particle.x, translate_noise)
    new_y = add_gaussian_noise(particle.y, translate_noise)
    new_h = add_gaussian_noise(particle.h, heading_noise)
    return Particle(new_x, new_y, new_h)

