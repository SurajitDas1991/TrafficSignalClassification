from pathlib import Path
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
#import os
import pickle as pickle
class PredictModel:
    def __init__(self):
        self.classes = { 0:'Speed limit (5km/h)',
            1:'Speed limit (15km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (40km/h)',
            4:'Speed limit (50km/h)',
            5:'Speed limit (60km/h)',
            6:'End of speed limit (70km/h)',
            7:'Speed limit (80km/h)',
            8:'Dont Go straight or left',
            9:'Dont Go straight or Right',
            10:'Dont Go straight',
            11:'Dont Go Left',
            12:'Dont Go Left or Right',
            13:'Dont Go Right',
            14:'Dont overtake from Left',
            15:'No Uturn',
            16:'No Car',
            17:'No horn',
            18:'Speed limit (40km/h)',
            19:'Speed limit (50km/h)',
            20:'Go straight or right',
            21:'Go straight',
            22:'Go Left',
            23:'Go Left or right',
            24:'Go Right',
            25:'keep Left',
            26:'keep Right',
            27:'Roundabout mandatory',
            28:'watch out for cars',
            29:'Horn',
            30:'Bicycles crossing',
            31:'Uturn',
            32:'Road Divider',
            33:'Traffic signals',
            34:'Danger Ahead',
            35:'Zebra Crossing',
            36:'Bicycles crossing',
            37:'Children crossing',
            38:'Dangerous curve to the left',
            39:'Dangerous curve to the right',
            40:'Unknown1',
            41:'Unknown2',
            42:'Unknown3',
            43:'Go right or straight',
            44:'Go left or straight',
            45:'Unknown4',
            46:'ZigZag Curve',
            47:'Train Crossing',
            48:'Under Construction',
            49:'Unknown5',
            50:'Fences',
            51:'Heavy Vehicle Accidents',
            52:'Unknown6',
            53:'Give Way',
            54:'No stopping',
            55:'No entry',
            56:'Unknown7',
            57:'Unknown8'
            }

    def predict_result(self,validation_generator):
        model = load_model('./models/trained_model.h5')
        score = model.evaluate(validation_generator)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def predict_uploaded_file(self,file_path):

        model = load_model('./models/trained_model.h5')
        img_array = np.array(file_path).astype(np.float32)
        img = tf.image.resize(img_array, size=(128,128))
        x = tf.expand_dims(img, axis=0)
        img_data = preprocess_input(x)
        predicted_class = model.predict(img_data)

        y_classes = predicted_class.argmax(axis=-1)
        #print(y_classes)
        #print(type(y_classes))
        item=y_classes.item(0)
        #print(item)

        class_item=0
        #Get the mapping
        filename=str(Path(__file__).resolve().parent)+'/map.pkl'
        #file_to_read = open(filename, "rb")
        # Load data (deserialize)
        with open(filename, 'rb') as handle:
            unserialized_data = pickle.load(handle)
        for key,value in unserialized_data.items():
            if item==key:
                class_item=value
                #print(f'final item is {class_item}')
                break
            #print(key," ",value)
        result = "Predicted Traffic🚦Sign is: " +str(self.classes.get(int(class_item)))
        print(result)
        return result
