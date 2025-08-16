# Heart Disease Classification

 Done By:
Akram Mohamed (KemoMoh11)</br>
Bassam Hassan (bassam1231)</br>
Mahmoud Maher (themastertm)</br>

## Overview
This repository contains a Heart Disease Classification project that aims to predict the presence of heart disease in patients using machine learning. The project leverages multiple machine learning models, including AdaBoost, Bagging, CatBoost, Extra Trees, Gradient Boosting, LightGBM, Logistic Regression, Random Forest, SVM, and XGBoost. The best-performing model is deployed as a web application using Flask and Joblib for model serialization.

## Key Features

### Multiple Machine Learning Models
The project explores and compares the performance of 10 different machine learning algorithms:
- AdaBoost
- Bagging
- CatBoost
- Extra Trees
- Gradient Boosting
- LightGBM
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

### Model Training and Evaluation
The dataset is preprocessed, and each model is trained and evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Model Deployment
- The best-performing model is serialized using Joblib and deployed as a web application using Flask.
- Users can input patient data through a web interface and get predictions on the likelihood of heart disease.

### User-Friendly Web Interface
- The Flask web application provides an intuitive interface for users to interact with the model and receive predictions.

## Technologies Used
- **Python**: Primary programming language
- **Flask**: Web framework for deploying the model
- **Joblib**: For serializing and loading the trained model
- **Scikit-learn**: For implementing machine learning models (AdaBoost, Bagging, Extra Trees, Gradient Boosting, Logistic Regression, Random Forest, SVM)
- **CatBoost**: For CatBoost classifier
- **LightGBM**: For LightGBM classifier
- **XGBoost**: For XGBoost classifier
- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical computations
- **Matplotlib/Seaborn**: For data visualization
- **HTML/CSS/JavaScript**: For the web interface

## How to Use

### Clone the Repository
```bash
git clone https://github.com/your-username/Heart-Disease-Classification.git
cd Heart-Disease-Classification
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Flask Application
```bash
cd app
python app.py
```

Open your browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the web application.

### Input Data and Get Predictions
Enter the required patient data in the web form and click "Predict" to get the heart disease classification result.

## Dataset
The dataset used in this project is the Heart Disease Dataset (e.g., from UCI Machine Learning Repository or Kaggle). It contains features such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol levels
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression induced by exercise
- Slope of the peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia
- Target (presence of heart disease: 0 = no, 1 = yes)


The best-performing model is selected based on these metrics and deployed.

## Future Improvements
- Add more advanced models like Neural Networks.
- Implement hyperparameter tuning for better performance.
- Add feature importance visualization.
- Deploy the application on a cloud platform (e.g., AWS, Heroku).

