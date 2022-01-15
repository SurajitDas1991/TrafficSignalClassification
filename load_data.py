from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os
import itertools as it, glob
import constants
import glob
from glob import glob

class DataLoader:
    def read_labels_of_dataset(self):
        labels_df = pd.read_csv("../labels.csv")
        return labels_df

    def multiple_file_types(self,*patterns):
        return it.chain.from_iterable(glob.iglob(pattern) for       pattern in patterns)

    def load_images_from_data_folder(self):
        image_list=[]
        exts = [".png", ".jpg", ".jpeg"]
        mainpath = str(Path(__file__).resolve().parent)+"\\traffic_Data\\DATA\\"
        print(mainpath)
        image_list = list([p for p in Path(mainpath).rglob("*") if p.suffix in exts])
        # for filename in self.multiple_file_types("*.png", "*.jpg", "*.jpeg"):
        #     image_list.append(list(Path("traffic_Data/DATA")+filename))
        #image_list = list(Path("traffic_Data/DATA").glob(patterns))
        labels = list(
            map(lambda path: os.path.split(os.path.split(path)[0])[1], image_list)
        )
        return (image_list, labels)

    def create_dataframe_of_images_and_labels(self, list_of_images, list_of_labels):
        #, labels = self.load_images_from_data_folder()
        image_series = pd.Series(list_of_images).astype(str)
        labels_series = pd.Series(list_of_labels).astype(str)
        frame = {"Filenames": image_series, "label": labels_series}
        image_df = pd.DataFrame(frame)
        self.visualize_data_labels_count(image_df)
        return image_df

    def visualize_data_labels_count(self, df):
        count_labels = df.groupby(["label"]).size()
        plt.figure(figsize=(17, 5))
        plt.ylabel("count images")
        sns.barplot(x=count_labels.index, y=count_labels, palette="rocket")
        plt.savefig("visualizations/label_count.png")

    def split_dataset(self, df):
        count_labels = df.groupby(["label"]).size()
        # print(count_labels)
        # print(type(count_labels))
        count_labels_df = count_labels.to_frame(name="count_images").reset_index()
        # count_labels_df.head()
        ## Drop rows less than the split minimum count
        drop_label_list = list(
            count_labels_df["label"].loc[
                count_labels_df["count_images"] < constants.SPLIT_MINIMUM_COUNT
            ]
        )

        drop_df = df.copy()
        split_df = df.copy()

        for index, row in df.iterrows():
            if str(row.label) in drop_label_list:
                split_df = split_df.drop(index)
            else:
                drop_df = drop_df.drop(index)

        return split_df, drop_df

    def custom_train_test_split(self, df):
        """
        Train test split where test_df has minimum 1 image in all labels
        in random split. This need to work model.fit and model.evaluate
        """

        labels = df.label.unique()
        test_df = pd.DataFrame()

        for label in labels:
            label_samples = df.loc[df.label == label]
            test_df = test_df.append(
                label_samples.sample(
                    len(label_samples) // 10 + 1, random_state=constants.SEED
                )
            )
        train_df = df.drop(list(test_df.index), axis=0)
        test_df = test_df.sample(frac=1, random_state=constants.SEED)
        train_df = train_df.sample(frac=1, random_state=constants.SEED)

        return train_df, test_df

    def visualize_train_data(self, df):
        # plot images
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(plt.imread(df.Filenames.iloc[idx]))
            ax.set_title(df.label.iloc[idx])
        plt.tight_layout()
        plt.savefig("visualizations/train.png")
        # plt.show()
