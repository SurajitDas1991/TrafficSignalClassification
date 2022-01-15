from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
from sklearn.utils import shuffle
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import math
import constants
import os
from pathlib import Path

class TrainModel:

    def train(self):
        batch_size=32
        datagen_args = dict(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode="constant",
            shear_range=0.1,
            zoom_range=0.2)
        datagen = ImageDataGenerator(**datagen_args)
        image_path=str(Path(__file__).resolve().parent)+'/traffic_Data/DATA'
        print(image_path)
        datagenerator = datagen.flow_from_directory(image_path,target_size=(128,128),batch_size=batch_size,interpolation="lanczos",shuffle=True)
        validation_generator = datagen.flow_from_directory(image_path,target_size=(128,128), batch_size=batch_size,interpolation="lanczos",shuffle=True)
        return datagenerator, validation_generator

    def train_generator(self):
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,

            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode="constant",
            shear_range=0.1,
            zoom_range=0.2
        )

        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,


        )

        return train_generator, test_generator

    def assign_data_from_dataset_to_generators(
        self, train_df, validation_df, test_df, train_generator, test_generator
    ):
        train_images = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col="Filenames",
            y_col='label',
            color_mode="rgb",
            class_mode="categorical",
            target_size=(128, 128),
            batch_size=constants.BATCH_SIZE,
            shuffle=True,
            seed=constants.SEED,
        )

        val_images = test_generator.flow_from_dataframe(
            dataframe=validation_df,
            x_col="Filenames",
            y_col='label',
            color_mode="rgb",
            class_mode="categorical",
            target_size=(128, 128),
            batch_size=constants.BATCH_SIZE,
            shuffle=True,
            seed=constants.SEED,
        )

        test_images = test_generator.flow_from_dataframe(
            dataframe=test_df,
            x_col="Filenames",
            color_mode="rgb",
            class_mode=None,
            seed=constants.SEED,
            shuffle=False,
            target_size=(128, 128),
            batch_size=constants.BATCH_SIZE,
        )

        return train_images, val_images, test_images

    def get_num_classes_in_train(self, datagenerator):
        #train_labels = df.groupby(["label"]).size()
        return len(datagenerator.class_indices)
        #return len(train_labels)

    # create model
    def create_model(self, num_classes, input_shape=(128, 128, 3)):

        inputs = tf.keras.layers.Input(input_shape)

        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            classifier_activation="softmax",
        )
        base_model.trainable = False

        x = base_model(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        return tf.keras.models.Model(inputs, outputs)

    def compile(self, model):
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit_model(self, model,datagenerator,validation_gen):
        batch_size=32
        callback = [tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        # tf.keras.callbacks.ModelCheckpoint('models/model_{val_accuracy:.3f}.h5',
        #                                 save_best_only=True,
        #                                 save_weights_only=False,
        #                                 monitor='val_accuracy')
        ]

        history=model.fit(datagenerator,
                          steps_per_epoch=int(np.ceil(datagenerator.samples / float(batch_size))),
                          validation_steps=int(np.ceil(validation_gen.samples / float(batch_size))),
                           epochs=25, verbose=1,callbacks=[callback],validation_data=validation_gen)
        print(history.history.keys())
        self.plot_train_test_accuracy(history)
        model_path=str(Path(__file__).resolve().parent)+'/models/trained_model.h5'
        model.save(model_path)
        class_mapping = {v:k for k,v in datagenerator.class_indices.items()}
        filename=str(Path(__file__).resolve().parent)+'/map.pkl'
        filehandler = open(filename, 'wb')
        pickle.dump(class_mapping, filehandler)
        return model

    def plot_train_test_accuracy(self, history):
        plt.figure(figsize=(12, 5))
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.title("Accuracy plot")
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("visualizations/accuracy.png")
        # plt.show()
