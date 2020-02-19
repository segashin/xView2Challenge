#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          # 
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, # 
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################


from PIL import Image
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import random
import argparse
import logging
import json
from sys import exit
import cv2
import datetime

import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict

import tensorflow as tf
import keras

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Lambda
from model_siamese import *

# Configurations
NUM_WORKERS = 4
NUM_CLASSES = 4
BATCH_SIZE = 64
NUM_EPOCHS = 120
LEARNING_RATE = 0.0001
RANDOM_SEED = 123
LOG_DIR = '/tmp/inference/classification_log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


damage_intensity_encoding = dict() 
damage_intensity_encoding[3] = 'destroyed' 
damage_intensity_encoding[2] = 'major-damage'
damage_intensity_encoding[1] = 'minor-damage'
damage_intensity_encoding[0] = 'no-damage'


###
# Creates data generator for validation set
###
def create_generator(test_df, test_dir, output_json_path):

    gen = keras.preprocessing.image.ImageDataGenerator(
                             rescale=1.4)

    try:
        gen_flow = gen.flow_from_dataframe(dataframe=test_df,
                                   directory=test_dir,
                                   x_col='uuid',
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   seed=RANDOM_SEED,
                                   class_mode=None,
                                   target_size=(128, 128))
    except:
        # No polys detected so write out a blank json
        blank = {}
        with open(output_json_path , 'w') as outfile:
            json.dump(blank, outfile)
        exit(0)


    return gen_flow

def generate_data_siamese(df_pre, df_post, in_dir_pre, in_dir_post):
    # df_pre = df_pre.replace({"labels" : damage_intensity_encoding })
    # df_post = df_post.replace({"labels" : damage_intensity_encoding })
    gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                             vertical_flip=True,
                                              width_shift_range=0.1,
                                           height_shift_range=0.1,
                                            rescale=1.4)
    gen_pre = gen.flow_from_dataframe(dataframe=df_pre,
                                   directory=in_dir_pre,
                                   x_col='uuid',
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   seed=RANDOM_SEED,
                                   class_mode=None,
                                   target_size=(128, 128))
    gen_post = gen.flow_from_dataframe(dataframe=df_post,
                                   directory=in_dir_post,
                                   x_col='uuid',
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   seed=RANDOM_SEED,
                                   class_mode=None,
                                   target_size=(128, 128))

    while True:
        pre_xy_i = gen_pre.next()
        post_xy_i = gen_post.next()
        # print("input shape", pre_xy_i.shape, post_xy_i.shape)
        
        # post_xy_i[1]'s shape is [BATCH_SIZE, NUM_CLASSES] and it is one-hot encoded
        # print("post_xy_i.shape", post_xy_i[1].shape)
        # print("label", post_xy_i[1].argmax(axis=1))
        # print("value", post_xy_i[1].argmax(axis=1) / 4 + 0.125)
        yield [pre_xy_i, post_xy_i]

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    # y_true = tf.Print(y_true, [y_true], "y_true", summarize=1000)
    # y_pred = tf.Print(y_pred, [y_pred], "y_pred", summarize=1000)

    # https://jdhao.github.io/2017/03/13/some_loss_and_explanations/
    margin = 0.125
    square_pred = tf.square(y_pred)
    # square_pred = K.square(y_pred)
    # square_pred = tf.Print(square_pred, [square_pred], "square_pred", summarize=1000)

    # margin_square = K.square(K.maximum(margin - y_pred, 0))
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    # margin_square = tf.Print(margin_square, [margin_square], "margin_square", summarize=1000)

    # mean = K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    mean = tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    # mean = tf.Print(mean, [mean], "mean", summarize=1000)

    return mean

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    # y_true = tf.Print(y_true, [y_true], "y_true", summarize=1000)
    # y_pred = tf.Print(y_pred, [y_pred], "y_pred", summarize=1000)

    # print(y_true)
    
    label_midpoints = tf.convert_to_tensor(np.array([0.125, 0.375, 0.625, 0.875]), dtype="float32")

    abs_dif = tf.math.abs(tf.math.subtract(y_pred, label_midpoints))
    # abs_dif = tf.Print(abs_dif, [abs_dif], "abs_dif", summarize=1000)

    tensor_indices = tf.argmin(abs_dif, axis=1)
    # tensor_indices = tf.Print(tensor_indices, [tensor_indices], "tensor_indices", summarize=1000)

    y_pred_nearest = tf.gather(label_midpoints, tensor_indices)
    # y_pred_nearest = tf.Print(y_pred_nearest, [y_pred_nearest], "y_pred_nearest", summarize=1000)
    
    y_true_flattened = tf.reshape(y_true, [-1])
    # y_true_flattened = tf.Print(y_true_flattened, [y_true_flattened], "y_true_flattened", summarize=1000)

    equal = tf.equal(y_true_flattened, y_pred_nearest)
    # equal = tf.Print(equal, [equal], "equal", summarize=1000)

    mean = tf.reduce_mean(tf.cast(equal, "float32"))
    # mean = tf.Print(mean, [mean], "mean", summarize=1000)

    return mean

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# Runs inference on given test data and pretrained model
def run_inference(test_data_pre, test_data_post, test_csv_pre, test_csv_post, model_weights, output_json_path):

    base_model = generate_xBD_baseline_model((128,128,3))

    input_a = Input(shape=(128,128,3)) 
    input_b = Input(shape=(128,128,3))

    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    model.load_weights(model_weights)

    adam = keras.optimizers.Adam(lr=LEARNING_RATE,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    decay=0.0,
                                    amsgrad=False)


    model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])

    df_pre = pd.read_csv(test_csv_pre)[:600]
    df_post = pd.read_csv(test_csv_post)[:600]

    test_gen = generate_data_siamese(df_pre, df_post, test_data_pre, test_data_post)
    # test_gen.reset()
    samples = df_pre["uuid"].count()

    steps = np.ceil(samples/BATCH_SIZE)

    tensorboard_callbacks = keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    predictions = model.predict_generator(generator=test_gen,
                    callbacks=[tensorboard_callbacks],
                    steps=steps,
                    verbose=1)

    # print("predictions:", predictions)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print("labels", df_post[:BATCH_SIZE * 10]["labels"])

    # convert predictions to 
    label_midpoints = np.array([0.125, 0.375, 0.625, 0.875])
    predictions_nearest = label_midpoints[abs(predictions.flatten()[None, :] - label_midpoints[:, None]).argmin(axis=0)]

    print(predictions_nearest)

    # calculate precision and recall for each class
    true_labels = df_post[:600]["labels"].to_numpy()
    print(true_labels)
    for damage_type in damage_intensity_encoding:
        print("Damage Type:", damage_type)
        total_positives = np.count_nonzero(np.isclose(predictions_nearest, damage_type/4+0.125))
        true_positives = np.count_nonzero(np.where(true_labels == damage_type, np.isclose(predictions_nearest, damage_type/4+0.125), 0))

        # true_positives = np.count_nonzero(np.extract(predictions_nearest))

        # print("total_positives: ", total_positives)
        # print("true_positives: ", true_positives)

        try:
            precision = true_positives / total_positives
        except:
            print(f"0 total_positives for {damage_type}")
            precision = 0

        total_damage_type_obs = np.count_nonzero(true_labels == damage_type)
        # print("total_damage_type_obs", total_damage_type_obs)

        recall = true_positives / total_damage_type_obs

        try:
            f1 = 2 * precision * recall / (precision + recall)
        except: 
            print(f"0 (precision + recall) for {damage_type}")
            f1 = 0

        print(f"{precision} {recall} {f1}")
    

    # predicted_indices = np.argmax(predictions, axis=1)
    # predictions_json = dict()
    # for i in range(samples):
    #     filename_raw = test_gen.filenames[i]
    #     filename = filename_raw.split(".")[0]
    #     predictions_json[filename] = damage_intensity_encoding[predicted_indices[i]]

    # with open(output_json_path , 'w') as outfile:
    #     json.dump(predictions_json, outfile)


def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--test_data_pre',
                        required=True,
                        metavar="/path/to/xBD_test_dir_pre",
                        help="Full path to the cropped pre-disaster dataset directory")
    parser.add_argument('--test_data_post',
                        required=True,
                        metavar="/path/to/xBD_test_dir_post",
                        help="Full path to the cropped post-disaster dataset directory")
    parser.add_argument('--test_csv_pre',
                        required=True,
                        metavar="/path/to/xBD_test_csv_pre",
                        help="Full path to the csv file with pre-disaster cropped building img filename")
    parser.add_argument('--test_csv_post',
                        required=True,
                        metavar="/path/to/xBD_test_csv_post",
                        help="Full path to the csv file with post-disaster cropped building img filename")
    parser.add_argument('--model_weights',
                        default=None,
                        metavar='/path/to/input_model_weights',
                        help="Path to input weights")
    parser.add_argument('--output_json',
                        required=True,
                        metavar="/path/to/output_json")

    args = parser.parse_args()

    run_inference(args.test_data_pre, args.test_data_post, args.test_csv_pre, args.test_csv_post, args.model_weights, args.output_json)


if __name__ == '__main__':
    main()
