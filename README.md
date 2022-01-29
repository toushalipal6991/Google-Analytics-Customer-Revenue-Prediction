#  Google-Analytics-Customer-Revenue-Prediction END-To-END Project Using Flask API & AWS :dollar:

## :cinema: Demo :point_down:

![Demo](https://github.com/toushalipal6991/Google-Analytics-Customer-Revenue-Prediction/blob/main/gacrp-prod.gif)

## Aim of this Project
- This project aims to predict revenue for a set of customers for a time period in the future.
- Different machine learning & deep learning techniques were used.
- *Deployment* was done using *Flask API on an AWS EC2 instance*.

## Datasets:-
- Source :arrow_right: [Kaggle](https://www.kaggle.com/c/ga-customer-revenue-prediction/data)

## :memo: HLD(High Level Design)
- Training dataset was created by leveraging the time-based angle in the way the data was provided by Kaggle.
- Extensive EDA & Feature Engineering and experimentation with different ML & DL models was done, which lead to a good RMSE of 0.88647 by the best model.

## Deployment using Flask API on AWS
- A simple web-app has been built using this model (as shown in the Demo).
- This web-app has also been ***Deployed into Production using Flask API on an AWS EC2 instance***.
- Using this web-app, you can upload a query file whch can take 1 or more data points and get the predicted reveneue in the UI.

## :file_folder: Libraries Used
:crayon: scikit-learn :crayon: XGBoost :crayon: Tensorflow :crayon: matplotlib :crayon: seaborn :crayon: numpy :crayon: pandas :crayon: prettytable :crayon: Flask

## :hammer_and_wrench: :toolbox: Tools and Softwares Used
- Google Colab
- Jupyter Notebook
- Sublime Text
- AWS

