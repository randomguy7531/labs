#!/usr/bin/env python3

'''
This is starter code for Lab 6 on Coordinate Frame transforms.

'''

import asyncio
import cozmo
import numpy as np
from cozmo.util import degrees

def get_relative_pose(object_pose, reference_frame_pose):
	#getting relative translation
	relative_position = np.subtract(object_pose.position, reference_frame_pose.position)
	#getting relative rotation(note: this is only working in z)
	relative_angle = object_pose.rotation.angle_z - reference_frame_pose.rotation.angle_z

	#this function would have been able to be written much easier if there was a straightforward way to go from a 4x4 rotation matrix to a pose
	#Cozmo's SDK makes it easy to go from a post to a 4x4 rotation matrix, but not the other way
	#if going the other way was available, then everything could be computed easily as a division on 4x4 rotation matrices

	return cozmo.util.pose_z_angle(relative_position.x, relative_position.y, relative_position.z, relative_angle)

def find_relative_cube_pose(robot: cozmo.robot.Robot):
	'''Looks for a cube while sitting still, prints the pose of the detected cube
	in world coordinate frame and relative to the robot coordinate frame.'''

	robot.move_lift(-3)
	robot.set_head_angle(degrees(0)).wait_for_completed()
	cube = None

	while True:
		try:
			cube = robot.world.wait_for_observed_light_cube(timeout=30)
			if cube:
				print("Robot pose: %s" % robot.pose)
				print("Cube pose: %s" % cube.pose)
				print("Cube pose in the robot coordinate frame: %s" % get_relative_pose(cube.pose, robot.pose))
		except asyncio.TimeoutError:
			print("Didn't find a cube")


if __name__ == '__main__':

	cozmo.run_program(find_relative_cube_pose)
