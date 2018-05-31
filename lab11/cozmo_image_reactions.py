#!/usr/bin/env python3

'''Make Cozmo behave like a Braitenberg machine with virtual light sensors and wheels as actuators.

The following is the starter code for lab.
'''

import asyncio
import time
import cozmo
import _pickle as cPickle
import numpy as np
import re
import random  
from imgclassification import *
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color, transform
from sklearn.neural_network import MLPClassifier


def load_classifier_from_disk(file):
    with open(file, 'rb') as fid:
        classifier = cPickle.load(fid) 
        return classifier
    return None

async def get_and_classify_image(robot, classifier):
    #get camera image
    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)
    image_features = classifier.extract_image_features([np.asarray(event.image)])
    image_prediction = classifier.predict_labels(image_features)
    return image_prediction

async def react_to_prediction(robot, prediction):
    if prediction == "drone":
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(-45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        robot.say_text("drone").wait_for_completed()
    elif prediction == "hands":
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(-45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(-45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        robot.say_text("hands").wait_for_completed()
    elif prediction == "inspection":
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        robot.say_text("inspection").wait_for_completed()
    elif prediction == "order":
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(-45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        robot.say_text("order").wait_for_completed()
    elif prediction == "place":
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(-45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(-45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        robot.say_text("place").wait_for_completed()
    elif prediction == "plane":
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(-45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(-45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(-45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        robot.say_text("plane").wait_for_completed()
    elif prediction == "truck":
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(45)).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        robot.say_text("truck").wait_for_completed()


async def cozmo_image_reactions(robot: cozmo.robot.Robot):
    print("loading classifier")
    classifier = load_classifier_from_disk("img_classifier_class.pkl")
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    while True:
        predictions = []
        num_predictions_window = 10
        for i in range(num_predictions_window):
            print("getting prediction")
            prediction = await get_and_classify_image(robot, classifier)
            print(prediction)
            predictions.append(prediction)
        # get the most common predictionout of the last N
        uniques, indices = np.unique(predictions, return_inverse=True)
        prediction = uniques[np.argmax(np.bincount(indices))]
        await react_to_prediction(robot, prediction)


if __name__ == "__main__":
    cozmo.run_program(cozmo_image_reactions, use_viewer=True, force_viewer_on_top=False)
