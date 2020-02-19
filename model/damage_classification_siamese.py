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
import cv2
import datetime
import sys


from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict

import tensorflow as tf
import keras
import ast
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Add, Input, Concatenate, Lambda
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from callbacks import CustomModelCheckpoint, CustomTensorBoard
import keras.losses
from keras import backend as K

from model_siamese import *

logging.basicConfig(level=logging.INFO)

# Configurations
NUM_WORKERS = 4 
NUM_CLASSES = 4
BATCH_SIZE = 64
NUM_EPOCHS = 100 
LEARNING_RATE = 0.0001
RANDOM_SEED = 123
LOG_STEP = 150
LOG_DIR = './logs_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

damage_intensity_encoding = dict()
damage_intensity_encoding[3] = '3'
damage_intensity_encoding[2] = '2' 
damage_intensity_encoding[1] = '1' 
damage_intensity_encoding[0] = '0' 

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

###
# Function to compute unweighted f1 scores, just for reference
###
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


###
# Creates data generator for validation set
###
def validation_generator(test_csv, test_dir):
    df = pd.read_csv(test_csv)
    df = df.replace({"labels" : damage_intensity_encoding })

    gen = keras.preprocessing.image.ImageDataGenerator(
                             rescale=1.4)


    return gen.flow_from_dataframe(dataframe=df,
                                   directory=test_dir,
                                   x_col='uuid',
                                   y_col='labels',
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   seed=RANDOM_SEED,
                                   class_mode="categorical",
                                   target_size=(128, 128))


def generate_validation_siamese(df, in_dir_pre, in_dir_post):
    df = df.replace({"labels" : damage_intensity_encoding })

    gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                             vertical_flip=True,
                                              width_shift_range=0.1,
                                           height_shift_range=0.1,
                                            rescale=1.4)
    gen_pre = gen.flow_from_dataframe(dataframe=df,
                                   directory=in_dir_pre,
                                   x_col='uuid',
                                   y_col='labels',
                                   batch_size=BATCH_SIZE,
                                   seed=RANDOM_SEED,
                                   class_mode="categorical",
                                   target_size=(128, 128))
    gen_post = gen.flow_from_dataframe(dataframe=df,
                                   directory=in_dir_post,
                                   x_col='uuid',
                                   y_col='labels',
                                   batch_size=BATCH_SIZE,
                                   seed=RANDOM_SEED,
                                   class_mode="categorical",
                                   target_size=(128, 128))

    while True:
        pre_xy_i = gen_pre.next()
        post_xy_i = gen_post.next()
        yield [pre_xy_i[0], post_xy_i[0]], post_xy_i[1] / 4 + 0.125

def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)

    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=5,
        mode='min',
        verbose=1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save=model_to_save,
        filepath=saved_weights_name,  # + '{epoch:02d}.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='min',
        epsilon=0.01,
        cooldown=0,
        min_lr=0
    )
    tensorboard = CustomTensorBoard(
        log_dir=tensorboard_logs,
        write_graph=True,
        write_images=True,
    )
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]


###
# Applies random transformations to training data
###
def augment_data(df, in_dir):

    df = df.replace({"labels" : damage_intensity_encoding })
    gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             rescale=1.4)
    return gen.flow_from_dataframe(dataframe=df,
                                   directory=in_dir,
                                   x_col='uuid',
                                   y_col='labels',
                                   batch_size=BATCH_SIZE,
                                   seed=RANDOM_SEED,
                                   class_mode="categorical",
                                   target_size=(128, 128))

def generate_data_siamese(df_pre, df_post, in_dir_pre, in_dir_post):
    df_pre = df_pre.replace({"labels" : damage_intensity_encoding })
    df_post = df_post.replace({"labels" : damage_intensity_encoding })
    gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                             vertical_flip=True,
                                              width_shift_range=0.1,
                                           height_shift_range=0.1,
                                            rescale=1.4)
    gen_pre = gen.flow_from_dataframe(dataframe=df_pre,
                                   directory=in_dir_pre,
                                   x_col='uuid',
                                   y_col='labels',
                                   batch_size=BATCH_SIZE,
                                   seed=RANDOM_SEED,
                                   class_mode="categorical",
                                   target_size=(128, 128))
    gen_post = gen.flow_from_dataframe(dataframe=df_post,
                                   directory=in_dir_post,
                                   x_col='uuid',
                                   y_col='labels',
                                   batch_size=BATCH_SIZE,
                                   seed=RANDOM_SEED,
                                   class_mode="categorical",
                                   target_size=(128, 128))

    while True:
        pre_xy_i = gen_pre.next()
        post_xy_i = gen_post.next()

        # print("shape", pre_xy_i[0].shape, post_xy_i[0].shape)
        # post_xy_i[1]'s shape is [BATCH_SIZE, NUM_CLASSES] and it is one-hot encoded
        # print("post_xy_i.shape", post_xy_i[1].shape)
        # print("label", post_xy_i[1].argmax(axis=1))
        # print("value", post_xy_i[1].argmax(axis=1) / 4 + 0.125)
        yield [pre_xy_i[0], post_xy_i[0]], post_xy_i[1].argmax(axis=1) / 4 + 0.125

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    # y_true = tf.Print(y_true, [y_true], "y_true", summarize=1000)
    # y_pred = tf.Print(y_pred, [y_pred], "y_pred", summarize=1000)

    # https://jdhao.github.io/2017/03/13/some_loss_and_explanations/
    margin = 0.001
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

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

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

def accuracy_loss(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    y_true = tf.Print(y_true, [y_true], "y_true", summarize=1000)
    y_pred = tf.Print(y_pred, [y_pred], "y_pred", summarize=1000)

    # print(y_true)
    
    label_midpoints = tf.convert_to_tensor(np.array([0.125, 0.375, 0.625, 0.875]), dtype="float32")

    abs_dif = tf.math.abs(tf.math.subtract(y_pred, label_midpoints))
    abs_dif = tf.Print(abs_dif, [abs_dif], "abs_dif", summarize=1000)

    tensor_indices = tf.argmin(abs_dif, axis=1)
    tensor_indices = tf.Print(tensor_indices, [tensor_indices], "tensor_indices", summarize=1000)

    y_pred_nearest = tf.gather(label_midpoints, tensor_indices)
    y_pred_nearest = tf.Print(y_pred_nearest, [y_pred_nearest], "y_pred_nearest", summarize=1000)
    
    y_true_flattened = tf.reshape(y_true, [-1])
    y_true_flattened = tf.Print(y_true_flattened, [y_true_flattened], "y_true_flattened", summarize=1000)

    equal = tf.equal(y_true_flattened, y_pred_nearest)
    equal = tf.Print(equal, [equal], "equal", summarize=1000)

    mean = tf.reduce_mean(tf.cast(equal, "float32"))
    mean = tf.Print(mean, [mean], "mean", summarize=1000)

    acc_loss = tf.divide(1, mean)

    return acc_loss

    # mean = K.mean(K.equal(y_true, K.cast(y_pred_nearest, y_true.dtype)))
    
    

    # return K.mean(K.equal(y_true, K.cast(y_pred_nearest, y_true.dtype)))
    # print("y_true.dtype",y_true.dtype)
    # return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# Run training and evaluation based on existing or new model
def train_model(train_data_pre, train_data_post, train_csv_pre, train_csv_post, test_data, test_csv, model_in, model_out):

    base_model = generate_xBD_baseline_model((128,128,3))

    input_a = Input(shape=(128,128,3)) 
    input_b = Input(shape=(128,128,3))

    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    #prediction = Dense(1, activation="sigmoid")(distance)

    model = Model([input_a, input_b], distance)

    # Add model weights if provided by user
    #if model_in is not None:
    #    model.load_weights(model_in)
    df_pre = pd.read_csv(train_csv_pre)
    df_post = pd.read_csv(train_csv_post)
    class_weights = compute_class_weight('balanced', np.unique(df_post['labels'].to_list()), df_post['labels'].to_list());
    d_class_weights = dict(enumerate(class_weights))

    # overwrite the key values with the midpoint of each classes range 
    temp = d_class_weights.copy()
    for key in temp:
        print(key)
        print(key/4+0.125)
        d_class_weights[key/4+0.125] = d_class_weights[key]
        del d_class_weights[key]

    samples = df_post['uuid'].count()
    steps = np.ceil(samples/BATCH_SIZE)

    train_generator = generate_data_siamese(df_pre, df_post, train_data_pre, train_data_post)

    # #Set up tensorboard logging
    # tensorboard_callbacks = keras.callbacks.TensorBoard(log_dir=LOG_DIR,
    #                                                     batch_size=BATCH_SIZE)

    
    # #Filepath to save model weights
    # filepath = model_out + "-saved-model-{epoch:02d}.hdf5"
    # checkpoints = keras.callbacks.ModelCheckpoint(filepath,
    #                                                 monitor=['loss'],
    #                                                 verbose=1,
    #                                                 save_best_only=False,
    #                                                 mode='max')

    callbacks = create_callbacks(saved_weights_name="siamese_v5.h5", tensorboard_logs=LOG_DIR, model_to_save=model)

    #Adds adam optimizer
    adam = keras.optimizers.Adam(lr=LEARNING_RATE,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    decay=0.0,
                                    amsgrad=False)


    model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])

    #Training begins
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps,
                        epochs=NUM_EPOCHS,
                        workers=NUM_WORKERS,
                        use_multiprocessing=True,
                        class_weight=d_class_weights,
                        callbacks=callbacks, #[tensorboard_callbacks, checkpoints],
                        verbose=1)


    # #Evalulate f1 weighted scores on validation set
    # validation_gen = generate_validation_siamese(test_csv, test_data)
    # predictions = model.predict(validation_gen)

    # val_trues = validation_gen.classes
    # val_pred = np.argmax(predictions, axis=-1)

    # f1_weighted = f1_score(val_trues, val_pred, average='weighted')
    # print(f1_weighted)

def validate(): 
    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--test_data_pre',
                        required=True,
                        metavar="/path/to/xBD_test",
                        help="Full path to the test data directory")
    parser.add_argument('--test_data_post',
                        required=True,
                        metavar="/path/to/xBD_test",
                        help="Full path to the test data directory")
    parser.add_argument('--test_csv',
                        required=True,
                        metavar="/path/to/xBD_split",
                        help="Full path to the test csv")

    args = parser.parse_args() 

    model = load_model('siamese_v2.h5', custom_objects={'contrastive_loss': contrastive_loss})

    #Evalulate f1 weighted scores on validation set
    df = pd.read_csv(args.test_csv)[0:100]
    num_data = df.shape[0]
    validation_gen = generate_validation_siamese(df, args.test_data_pre, args.test_data_post)
    predictions = model.predict(validation_gen, steps=num_data // BATCH_SIZE, verbose=1)

    df = df.replace({"labels" : damage_intensity_encoding })
    val_trues = df["labels"].values #validation_gen.classes
    print(val_trues.shape)
    print(predictions[:30, :])
    val_pred = np.argmax(predictions, axis=-1)
    print(val_pred.shape)

    f1_weighted = f1_score(val_trues, val_pred, average='weighted')
    print(f1_weighted)


def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--train_data_pre',
                        required=True,
                        metavar="/path/to/xBD_train",
                        help="Full path to the train data directory")
    parser.add_argument('--train_data_post',
                        required=True,
                        metavar="/path/to/xBD_train",
                        help="Full path to the train data directory")
    parser.add_argument('--train_csv_pre',
                        required=True,
                        metavar="/path/to/xBD_split",
                        help="Full path to the pre-disaster train csv")
    parser.add_argument('--train_csv_post',
                        required=True,
                        metavar="/path/to/xBD_split",
                        help="Full path to the post-disaster train csv")
    parser.add_argument('--test_data',
                        required=True,
                        metavar="/path/to/xBD_test",
                        help="Full path to the test data directory")
    parser.add_argument('--test_csv',
                        required=True,
                        metavar="/path/to/xBD_split",
                        help="Full path to the test csv")
    parser.add_argument('--model_in',
                        default=None,
                        metavar='/path/to/input_model',
                        help="Path to save model")
    parser.add_argument('--model_out',
                        required=True,
                        metavar='/path/to/save_model',
                        help="Path to save model")

    args = parser.parse_args()

    train_model(args.train_data_pre, args.train_data_post, args.train_csv_pre, args.train_csv_post, args.test_data, args.test_csv, args.model_in, args.model_out)


if __name__ == '__main__':
    main()
    # validate()
