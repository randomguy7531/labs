#!/usr/bin/env python

##############
#### Your name:
##############
import _pickle as cPickle
import numpy as np
import re
import random 
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color, transform
from sklearn.neural_network import MLPClassifier

class ImageClassifier:
    
    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)
        
        #create one large array of image data
        data = io.concatenate_images(ic)
        
        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]
        
        return(data,labels)

    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block):
        features = feature.hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=False, feature_vector=True)
        return features

    def bin_spatial(self, img, size=(120, 160)):
        features = transform.resize(img, size).ravel()
        return features

    def extract_single_image_features(self, single_image):
        hog_features = []
        orientations = 13
        pix_per_cell = 4
        cell_per_block = 2
        hog_features = self.get_hog_features(single_image, orientations, pix_per_cell, cell_per_block)
        spatial_features = self.bin_spatial(single_image)
        feature_data =  np.hstack(( hog_features, spatial_features))
        return feature_data

    def extract_image_features(self, data):
        feature_data = []
        for image in data:
            image_grey = color.rgb2grey(image)
            image_features = self.extract_single_image_features(image_grey)
            feature_data.append(image_features)        
        # Please do not modify the return type below
        return(feature_data)

    def rotate_training_image(self, image, angle):
        return transform.rotate(image, angle)
    
    def create_augmented_rotation(self, image, angle_range=(-5,5)):
        pertubation_angle = random.randrange(angle_range[0], angle_range[1])
        return self.rotate_training_image(image, pertubation_angle)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above

        clf = svm.LinearSVC(C=0.1)
        clf.fit(train_data, train_labels)
        self.classifier = clf

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels
        
        predicted_labels = self.classifier.predict(data)
        
        # Please do not modify the return type below
        return predicted_labels

    def save_classifier_to_disk(self, file):
        with open(file, 'wb') as fid:
            cPickle.dump(self.classifier, fid)   
      
def main():

    img_clf = ImageClassifier()

    # load images
    print("\nLoading images.....")
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    
    # convert images into features
    print("\nComputing image features.....")
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)
    
    # train model and test on training data
    print("\nTraining the classifier.....")
    img_clf.train_classifier(train_data, train_labels)
    print("\nSaving the classifier to disk.....")
    img_clf.save_classifier_to_disk("classifier_model.pkl")
    print("\nPredicting the results.....")
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))
    
    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))

    with open("img_classifier_class.pkl", 'wb') as fid:
        cPickle.dump(img_clf, fid)  

if __name__ == "__main__":
    main()
