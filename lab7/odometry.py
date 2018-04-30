#!/usr/bin/env python3

'''
Stater code for Lab 7.

'''

import cozmo
from cozmo.util import degrees, Angle, Pose, distance_mm, speed_mmps
import numpy as np
import math
import time

# Wrappers for existing Cozmo navigation functions

def cozmo_drive_straight(robot, dist, speed):
	"""Drives the robot straight.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		dist -- Desired distance of the movement in millimeters
		speed -- Desired speed of the movement in millimeters per second
	"""
	robot.drive_straight(distance_mm(dist), speed_mmps(speed)).wait_for_completed()

def cozmo_turn_in_place(robot, angle, speed):
	"""Rotates the robot in place.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		angle -- Desired distance of the movement in degrees
		speed -- Desired speed of the movement in degrees per second
	"""
	robot.turn_in_place(degrees(angle), speed=degrees(speed)).wait_for_completed()

def cozmo_go_to_pose(robot, x, y, angle_z):
	"""Moves the robot to a pose relative to its current pose.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		x,y -- Desired position of the robot in millimeters
		angle_z -- Desired rotation of the robot around the vertical axis in degrees
	"""
	robot.go_to_pose(Pose(x, y, 0, angle_z=degrees(angle_z)), relative_to_robot=True).wait_for_completed()

# Functions to be defined as part of the labs

def get_front_wheel_radius():
	"""Returns the radius of the Cozmo robot's front wheel in millimeters."""

	# radius was determined empirically by observing that moving forward between 86 and 87 mm results in
	# one full rotation of the front wheel.
	# therefore: 86.5/pi/2 ~= 13.77

	# interestingly, hand measurment of the radius with calipers yielded a measurement of 14mm. It is possible that
	# the control loop in Cozmo's drive_straight(...) function is not as exact/well tuned as it should be
	return 13.77

def get_distance_between_wheels():
	"""Returns the distance between the wheels of the Cozmo robot in millimeters."""
	# distance was determined emperically by observing that a 110 degree rotation in place 
	# resulted in the front wheels making one complete turn. 
	# Therefore: D = (2 * pi * R_wheel * 360deg) / (110deg * 2 * pi) ~= 45mm 
	return 45

def rotate_front_wheel(robot, angle_deg):
	"""Rotates the front wheel of the robot by a desired angle.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		angle_deg -- Desired rotation of the wheel in degrees
	"""
	# we can get the forward distance the wheel needs to turn from the angle
	# distance = 2 * pi * radius * (angle_deg / 360.0)
	cozmo_drive_straight(robot, 2 * math.pi * get_front_wheel_radius() * (angle_deg / 360.0), 50)

def my_drive_straight(robot, dist, speed):
	"""Drives the robot straight.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		dist -- Desired distance of the movement in millimeters
		speed -- Desired speed of the movement in millimeters per second
	"""
	# 0.65 is a constant offset to deal with the acceleration/ramp that is built in to the underlying speed controller
	duration = ((dist / speed) + 0.65)
	robot.drive_wheels(speed, speed, duration=duration)
	time.sleep(duration)

def my_turn_in_place(robot, angle, speed):
	"""Rotates the robot in place.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		angle -- Desired distance of the movement in degrees
		speed -- Desired speed of the movement in degrees per second
	"""
	tangential_dist = 2 * math.pi * get_distance_between_wheels() * abs(angle) / 360.0
	drive_duration = (tangential_dist / speed) + 0.65
	robot.drive_wheels(-1.0 * math.copysign(speed, angle), math.copysign(speed, angle), duration = drive_duration)
	time.sleep(drive_duration)
	pass

def my_go_to_pose1(robot, x, y, angle_z):
	"""Moves the robot to a pose relative to its current pose.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		x,y -- Desired position of the robot in millimeters
		angle_z -- Desired rotation of the robot around the vertical axis in degrees
	"""
	#first, get the current robot pose and its current x,y,angle_z
	robot_pose = robot.pose
	robot_x = robot_pose.position.x
	robot_y = robot_pose.position.y
	robot_angle_z = robot_pose.rotation.angle_z.degrees

	#redefine targets in world coordinates
	x = robot_pose.position.x + x
	y = robot_pose.position.y + y
	angle_z = robot_angle_z + angle_z
	
	#compute the heading we need in order to go to the destination (x,y)
	desired_heading_radians = math.atan2(y - robot_y, x - robot_x)
	desired_heading_degrees = (360 * desired_heading_radians) / (2 * math.pi)

	# calculate the relative angle we need to turn to reach our desired heading, and go there
	degrees_to_turn = desired_heading_degrees - robot_angle_z
	
	print(str(robot_x) + "," + str(robot_y) + "    " + str(robot_angle_z))
	print(str(degrees_to_turn))
	my_turn_in_place(robot, degrees_to_turn, 50)
	robot.stop_all_motors()
	# compute the forward distance we need in order to reach the destination (x,y) 
	# then, go there
	desired_distance_mm = math.sqrt(((x - robot_x) ** 2) + ((y - robot_y) ** 2))
	print(str(desired_distance_mm))
	my_drive_straight(robot, desired_distance_mm, 50)
	robot.stop_all_motors()
	# next, compute the relative angle we need to turn to reach our desired ending pose orientation
	# then, turn there
	degrees_to_turn = angle_z - robot.pose.rotation.angle_z.degrees
	print(str(degrees_to_turn))
	my_turn_in_place(robot, degrees_to_turn, 50)
	robot.stop_all_motors()

def GetLineABC(x,y,alpha):
	if(alpha == 90.0 or alpha == 270.0):
		return 1.0, 0.0, x
	else:
		slope = math.sin(math.radians(alpha)) / math.cos(math.radians(alpha))
		b = y - (x * slope)
		print(str(x) + "," + str(y) + "," + str(alpha) + "," + str(slope) +"," +  str(b))
		return -1.0 * slope, 1.0, b

def Dist2d(x1,y1,x2,y2):
	return math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)

def GetAngle(left, right, center):
	leg_1_len = Dist2d(left[0], left[1], center[0],center[1])
	leg_2_len = Dist2d(right[0], right[1], center[0], center[1])
	return math.degrees(math.atan(leg_2_len/leg_1_len))

def Constrain180(angle):
	if(angle > 180):
		return Constrain180(angle - 360)
	elif(angle < -180):
		return Constrain180(angle + 360)
	else:
		return angle

def my_go_to_pose2(robot, x, y, angle_z):
	"""Moves the robot to a pose relative to its current pose.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		x,y -- Desired position of the robot in millimeters
		angle_z -- Desired rotation of the robot around the vertical axis in degrees
	"""
	#redefine targets in world coordinates
	x = robot.pose.position.x + x
	y = robot.pose.position.y + y
	angle_z = robot.pose.rotation.angle_z.degrees + angle_z

	#algo taken from the Correl book (chapter 3.5)
	while True:
		
		x_error = x - robot.pose.position.x
		y_error = y - robot.pose.position.y
		# rho represents our current straight line distance error to the goal
		rho = math.sqrt((x_error ** 2) + (y_error ** 2))
		# alpha represents the delta between our current heading, and the heading we need to be taking to reach the goal
		alpha = Constrain180(robot.pose.rotation.angle_z.degrees - math.degrees(math.atan2(y_error, x_error)))
		# eta represents the delta between our final desired heading, and our current one
		eta = Constrain180(robot.pose.rotation.angle_z.degrees - angle_z)

		# our gains for the controller
		p1 = 0.15 # p1 is the proportional gain on the current rho error
		rho_cieling = 25.0 #we will linearly schedule the p2 and p3 gains based upon our current rho error
		p2_p3_weight = min(rho / rho_cieling, 1.0) # essentially, the closer we are to the goal, the higher we weight the eta error vs the alpha error
		p2 = (p2_p3_weight) * -0.02 # p2 is the proportional gain on the current alpha error
		p3 = (1.0 - p2_p3_weight) * -0.02 # p3 is the proportional gain on the current eta error

		translation_component = min(p1 * rho, 25) #cap errors to avoid shenanigans
		rotation_component = p2 * alpha + p3 * eta
		#compute the naive rotational motor inputs and then simply add them to the naive translation component
		motor_adjustments = rotation_component * get_distance_between_wheels() / 2 
		left_motor = translation_component - motor_adjustments
		right_motor = translation_component + motor_adjustments
		robot.drive_wheel_motors(left_motor, right_motor)

		time.sleep(0.1)

		# when we get close the the target, stop moving
		if(rho < 20 and abs(eta) < 5):
			robot.stop_all_motors()
			return

def my_go_to_pose3(robot, x, y, angle_z):
	"""Moves the robot to a pose relative to its current pose.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		x,y -- Desired position of the robot in millimeters
		angle_z -- Desired rotation of the robot around the vertical axis in degrees
	"""
	#redefine targets in world coordinates
	x = robot.pose.position.x + x
	y = robot.pose.position.y + y
	angle_z = robot.pose.rotation.angle_z.degrees + angle_z

	x_error = x - robot.pose.position.x
	y_error = y - robot.pose.position.y
	# same error calculation as from the pose2 method 
	alpha = Constrain180(robot.pose.rotation.angle_z.degrees - math.degrees(math.atan2(y_error, x_error)))
	if(alpha > 90 or alpha < -90):
		# target must be behind robot - do method 1
		my_go_to_pose1(robot, x, y, angle_z)
	else:
		my_go_to_pose2(robot, x, y, angle_z)

def run(robot: cozmo.robot.Robot):

	print("***** Front wheel radius: " + str(get_front_wheel_radius()))
	print("***** Distance between wheels: " + str(get_distance_between_wheels()))

	## Example tests of the functions

	#cozmo_drive_straight(robot, 86, 50)
	#cozmo_turn_in_place(robot, 110, 30)
	#cozmo_go_to_pose(robot, 100, 100, 45)

	#rotate_front_wheel(robot, 180)
	#my_drive_straight(robot,50, 50)
	#my_turn_in_place(robot, 90, 30)

	#my_go_to_pose1(robot, 100, 100, 180)
	#my_go_to_pose2(robot, 100, 100, 45)
	my_go_to_pose3(robot, 100, 100, 45)


if __name__ == '__main__':

	cozmo.run_program(run)



