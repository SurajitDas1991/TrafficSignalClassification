import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

from load_data import DataLoader
from train import TrainModel
from predict import PredictModel
from dashboard import PrepareTheDasboard



if __name__ == "__main__":
    dasboard_obj=PrepareTheDasboard()
    dasboard_obj.get_ready()
    # data_load_object = DataLoader()
    # #label_df = data_load_object.read_labels_of_dataset()
    # image_list, labels = data_load_object.load_images_from_data_folder()
    # image_df = data_load_object.create_dataframe_of_images_and_labels(
    #     image_list, labels
    # )
    # split_df, drop_df = data_load_object.split_dataset(image_df)
    # train_df, test_df = data_load_object.custom_train_test_split(split_df)
    # train, val = data_load_object.custom_train_test_split(train_df)
    # data_load_object.visualize_train_data(train)
    # label_map = dict(train.values)
    # # Train setup
    # train_object = TrainModel()
    # train_generator, validation_generator = train_object.train()
    # num_classes = train_object.get_num_classes_in_train(train_generator)
    # # (
    # #     train_images,
    # #     val_images,
    # #     test_images,
    # # ) = train_object.assign_data_from_dataset_to_generators(
    # #     train, val, test_df, train_generator, validation_generator
    # # )
    # model = train_object.create_model(num_classes)
    # compiled_model = train_object.compile(model)
    # #fitted_model = train_object.fit_model(compiled_model, train_images, val_images)
    # fitted_model = train_object.fit_model(compiled_model,train_generator,validation_generator)
    predict_object = PredictModel()

    #predict_object.predict_result(validation_generator)
    #predict_object.predict(fitted_model, test_images,test_generator)
    #predict_object.predict(fitted_model, test_images,test_generator)
