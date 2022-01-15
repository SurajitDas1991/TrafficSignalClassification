import streamlit as st
#import pathlib
#import os
from PIL import Image
#import matplotlib.pyplot as plt

from predict import PredictModel

class PrepareTheDasboard:

    @st.cache
    def load_image(self,image_file):
        img = Image.open(image_file)
        return img
    def get_ready(self):
        predict_obj=PredictModel()
        st.header("TRAFFIC SIGNAL CLASSIFICATION")
        uploaded_file=st.file_uploader(label='Upload traffic signals to know the meaning',type=['png','jpg','jpeg'])

        if uploaded_file is not None:
                if uploaded_file.name.find('.png') or uploaded_file.name.find('.jpg')or uploaded_file.name.find('.jpeg'):
                    st.success(f'Valid File - {uploaded_file.name}')
                image = Image.open(uploaded_file)
                file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type,
                              "filesize":uploaded_file.size}
                #st.write(file_details)
                #st.image(self.load_image(uploaded_file))
                st.image(
                        image,
                        caption=f"Uploaded image",
                        use_column_width=True,
    )

        if st.button("Predict Signal Type "):

                # size=(20,20)
                # image.thumbnail(size)
                # fig = plt.figure()
                # plt.imshow(image)
                # plt.axis("off")
                # st.pyplot(fig)
                # with open(os.path.join("uploadedimages",uploaded_file.name),"wb") as f:
                #     f.write((uploaded_file).getbuffer())
                #st.success("File saved")
                # st.write(predict_obj.predict_uploaded_file(os.path.join("uploadedimages",uploaded_file.name)))

                st.write(predict_obj.predict_uploaded_file(Image.open(uploaded_file)))
