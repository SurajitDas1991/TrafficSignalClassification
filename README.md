
# TRAFFIC SIGN CLASSIFIER


### APP 

- LINK - https://trafficsignalprediction.herokuapp.com/

### APP OUTPUT


### DATA INFO
- There are around 58 classes and each class has around 120 images. the labels.csv file has the respective description of the traffic sign class.

### PROJECT GOAL
- Create a basic image classifier with Efficient Net architecture to classifiy traffic signals. No hyperparameter tuning has been done as basic model ( EfficentNetB0 ) gave around 97% validation accuracy 
- Build a Streamlit UI to provide an visual look and feel
- Deploy to AWS Azure platform







## Deployment

To deploy this project run

```bash
  stream run app.py
```


## Run Locally

Clone the project

```bash
  git clone https://github.com/SurajitDas1991/TrafficSignalClassification.git
```

Go to the project directory

```bash
  cd "Traffic Sign Detection"
```

Install dependencies

```bash
  pip install - r requirements.txt
```

Start the server

```bash
  streamlit run app.py
```

