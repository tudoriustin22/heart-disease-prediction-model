# Heart Disease Prediction Model

![Heart Disease Prediction Model Thumbnail](Heart%20Disease%20Prediction%20Model%20Thumbnail.png)

A complete machine learning pipeline: from data preprocessing to production API using XGBoost and FastAPI

---

## Abstract

This project presents a machine learning system for predicting heart disease risk based on clinical and physiological features. The model employs an XGBoost classifier optimized through grid search cross-validation, achieving 87.5% accuracy on the test dataset. The system is deployed as a production-ready REST API using FastAPI, providing real-time predictions with comprehensive validation and error handling.

## 1. Introduction

Cardiovascular diseases remain a leading cause of mortality worldwide. Early detection and risk assessment can significantly improve patient outcomes through timely intervention. This work implements a machine learning-based prediction system that analyzes patient clinical data to assess heart disease risk.

The system follows a complete machine learning pipeline, from exploratory data analysis and preprocessing to model training, evaluation, and deployment as a scalable web service.

## 2. Dataset

The dataset used in this study is the Heart Disease Dataset, available on Kaggle:
- **Source**: [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Format**: CSV
- **Features**: 11 clinical and physiological features
- **Target**: Binary classification (presence/absence of heart disease)

### 2.1 Feature Description

The dataset contains the following features:

- **Age**: Patient age in years
- **Sex**: Biological sex (M/F)
- **ChestPainType**: Type of chest pain (ATA, NAP, ASY, TA)
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol level (mg/dl)
- **FastingBS**: Fasting blood sugar > 120 mg/dl (binary: 0/1)
- **RestingECG**: Resting electrocardiographic results (Normal, ST, LVH)
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (Y/N)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: Slope of the peak exercise ST segment (Up, Flat, Down)

### 2.2 Data Preprocessing

The preprocessing pipeline includes:

1. **Numerical Features**: Standardized using `StandardScaler` from scikit-learn
   - Features: Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak

2. **Categorical Features**: Encoded using `OneHotEncoder` with first category drop
   - Features: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope

3. **Preprocessing Pipeline**: Implemented using `ColumnTransformer` for seamless integration of scaling and encoding operations

4. **Train-Test Split**: 80-20 split with random state 42 for reproducibility

## 3. Methodology

### 3.1 Model Selection

XGBoost (Extreme Gradient Boosting) was selected as the primary algorithm due to its:
- High performance on structured/tabular data
- Built-in regularization to prevent overfitting
- Ability to handle mixed data types effectively
- Robustness to missing values and outliers

### 3.2 Hyperparameter Optimization

Grid Search Cross-Validation (GridSearchCV) was employed to optimize hyperparameters:

- **Cross-validation**: 5-fold
- **Scoring metric**: Accuracy
- **Search space**: 3,888 parameter combinations
- **Parallel processing**: All CPU cores utilized (n_jobs=-1)

**Hyperparameter Grid**:
- `n_estimators`: [50, 100, 200]
- `learning_rate`: [0.01, 0.05, 0.1, 0.3]
- `max_depth`: [3, 5, 7, 9]
- `min_child_weight`: [1, 3, 5]
- `subsample`: [0.6, 0.8, 1.0]
- `colsample_bytree`: [0.6, 0.8, 1.0]
- `gamma`: [0, 0.1, 0.3]

### 3.3 Optimal Parameters

The best parameters identified through grid search:
- `colsample_bytree`: 0.8
- `gamma`: 0
- `learning_rate`: 0.05
- `max_depth`: 5
- `min_child_weight`: 5
- `n_estimators`: 200
- `subsample`: 0.8

**Best Cross-Validation Score**: 88.15%

## 4. Results

### 4.1 Model Performance

The optimized XGBoost model achieved the following performance metrics:

- **Test Accuracy**: 87.5%
- **Precision (Class 0)**: 0.85
- **Recall (Class 0)**: 0.86
- **F1-Score (Class 0)**: 0.85
- **Precision (Class 1)**: 0.90
- **Recall (Class 1)**: 0.89
- **F1-Score (Class 1)**: 0.89

### 4.2 Classification Report

```
              precision    recall  f1-score   support

           0       0.85      0.86      0.85        77
           1       0.90      0.89      0.89       107

    accuracy                           0.88       184
   macro avg       0.87      0.87      0.87       184
weighted avg       0.88      0.88      0.88       184
```

## 5. Model Deployment

### 5.1 API Architecture

The model is deployed as a REST API using FastAPI, a modern, fast web framework for building APIs with Python. Key features:

- **Automatic API documentation**: Interactive Swagger UI and ReDoc
- **Data validation**: Pydantic models for request/response validation
- **Type hints**: Full Python type support
- **Async support**: Built on Starlette and Pydantic
- **Production-ready**: ASGI server (Uvicorn) with high performance

### 5.2 API Endpoints

#### Root Endpoint
- **GET** `/`: API information and available endpoints

#### Health Check
- **GET** `/health`: Service health status and model loading verification

#### Single Prediction
- **POST** `/predict`: Predict heart disease risk for a single patient
  - **Request**: JSON object with patient features
  - **Response**: Prediction, probability, risk level, and confidence score

#### Batch Prediction
- **POST** `/predict/batch`: Predict heart disease risk for multiple patients
  - **Request**: Array of patient objects
  - **Response**: Array of predictions with patient IDs

### 5.3 Request Schema

The API accepts patient data with the following schema:

```json
{
  "Age": 54,
  "Sex": "M",
  "ChestPainType": "NAP",
  "RestingBP": 150,
  "Cholesterol": 195,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 122,
  "ExerciseAngina": "N",
  "Oldpeak": 0.0,
  "ST_Slope": "Up"
}
```

### 5.4 Response Schema

The API returns predictions in the following format:

```json
{
  "prediction": 0,
  "has_heart_disease": false,
  "risk_level": "Low",
  "probability": {
    "no_disease": 0.85,
    "has_disease": 0.15
  },
  "confidence": 0.85,
  "message": "Low risk of heart disease. Continue healthy lifestyle."
}
```

### 5.5 Risk Level Classification

The system classifies risk levels based on disease probability:
- **Low Risk**: Probability < 0.3
- **Moderate Risk**: 0.3 ≤ Probability < 0.6
- **High Risk**: Probability ≥ 0.6

## 6. Installation and Setup

### 6.1 Prerequisites

- Python 3.8 or higher
- pip package manager

### 6.2 Dependencies

Install required packages:

```bash
pip install fastapi uvicorn pydantic xgboost scikit-learn pandas numpy joblib
```

### 6.3 Required Files

Ensure the following files are present in the project directory:
- `heart_disease_xgb_model.json`: Trained XGBoost model
- `preprocessor.pkl`: Fitted preprocessing pipeline
- `deployment_inference.py`: API server script

## 7. Usage

### 7.1 Starting the API Server

Run the following command to start the API server:

```bash
python deployment_inference.py
```

The API will be available at `http://localhost:8000`

### 7.2 API Documentation

Once the server is running, access the interactive API documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 7.3 Making Predictions

#### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 54,
       "Sex": "M",
       "ChestPainType": "NAP",
       "RestingBP": 150,
       "Cholesterol": 195,
       "FastingBS": 0,
       "RestingECG": "Normal",
       "MaxHR": 122,
       "ExerciseAngina": "N",
       "Oldpeak": 0.0,
       "ST_Slope": "Up"
     }'
```

#### Using Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "Age": 54,
    "Sex": "M",
    "ChestPainType": "NAP",
    "RestingBP": 150,
    "Cholesterol": 195,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 122,
    "ExerciseAngina": "N",
    "Oldpeak": 0.0,
    "ST_Slope": "Up"
}

response = requests.post(url, json=data)
print(response.json())
```

## 8. Model Persistence

The trained model and preprocessing pipeline are saved in multiple formats:

1. **XGBoost JSON**: `heart_disease_xgb_model.json` - Native XGBoost format for efficient loading
2. **Pickle**: `heart_disease_xgb_model.pkl` - Python pickle format for compatibility
3. **Preprocessor**: `preprocessor.pkl` - Joblib serialized preprocessing pipeline

## 9. Technology Stack

### 9.1 Machine Learning
- **XGBoost**: Gradient boosting framework for model training
- **scikit-learn**: Preprocessing, model selection, and evaluation
- **NumPy**: Numerical computations
- **pandas**: Data manipulation and analysis

### 9.2 API Development
- **FastAPI**: Modern web framework for API development
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server for production deployment

### 9.3 Model Serialization
- **joblib**: Efficient serialization of scikit-learn objects
- **pickle**: Python object serialization

### 9.4 Development Environment
- **Jupyter Notebook**: Interactive development and experimentation

## 10. Project Structure

```
Heart Disease Model/
│
├── HeartDiseaseDetectionModel.ipynb    # Model training notebook
├── deployment_inference.py              # FastAPI deployment script
├── heart.csv                            # Dataset
├── heart_disease_xgb_model.json         # Trained model (XGBoost format)
├── heart_disease_xgb_model.pkl          # Trained model (Pickle format)
├── preprocessor.pkl                     # Preprocessing pipeline
├── Heart Disease Prediction Model Thumbnail.png  # Project thumbnail
└── README.md                            # This file
```

## 11. Limitations and Future Work

### 11.1 Limitations

- Model performance is dependent on data quality and feature availability
- The model is trained on a specific dataset and may not generalize to all populations
- Clinical validation with medical professionals is recommended before clinical use
- The current implementation does not include model versioning or A/B testing

### 11.2 Future Improvements

- Integration with electronic health record (EHR) systems
- Real-time monitoring and model performance tracking
- Model interpretability using SHAP values or LIME
- Expanded feature engineering and feature selection
- Ensemble methods combining multiple algorithms
- Deployment to cloud platforms (AWS, Azure, GCP) with containerization
- Implementation of model versioning and continuous learning

## 12. References

### 12.1 Libraries and Frameworks

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
- Ramírez, S. (2018). FastAPI: Modern, Fast, Web Framework for Building APIs with Python.

### 12.2 Dataset

- Heart Disease Dataset. (n.d.). Kaggle. Retrieved from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

## 13. License

This project is provided for educational and research purposes. Please ensure compliance with dataset licensing and medical data regulations when using this code in clinical or commercial applications.

## 14. Contact

For questions, issues, or contributions, please refer to the project repository or contact the maintainer.

---

**Note**: This model is intended for research and educational purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

