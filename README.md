# Symptom → Disease Predictor

**A machine learning–powered web application that predicts probable diseases based on user-entered symptoms.**

---

## 🚀 Project Overview

This application allows users to predict potential diseases from symptoms using multiple models.  
It supports **single-record JSON input** and **bulk CSV uploads**, returning interactive results or downloadable predictions.

✨ **Features**
- Evaluation of multiple classifiers (Decision Tree, RandomForest, NaiveBayes, SVM, Logistic Regression).  
- Flask-based UI for single and bulk predictions.  
- Downloadable CSV output for CSV-based predictions.  
- Modular, CLI-compatible code organization with logging and error handling.  

---

## 🖼️ Screenshot

![App UI screenshot](E:\HACKATHON MEDICAL SOLUTION\image.png)  
_Replace the above path with your actual UI screenshot for better presentation._

---

## 📑 Table of Contents

- [Installation](#installation)  
- [Usage](#usage)  
  - [Train the Model](#train-the-model)  
  - [Run the App](#run-the-app)  
  - [Make Predictions](#make-predictions)  
- [Project Structure](#project-structure)  
- [Model Training Details](#model-training-details)  
- [Tech Stack](#tech-stack)  
- [License & Contact](#license--contact)  

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MuzahidulIslam1/Hackathon-2.git
   cd Hackathon-2
2. **Create and activate a virtual environment**
   conda create venv python==3.9 -y
   conda activate venv/
3. **Install dependencies**
   pip install -r requirements.txt

## ▶️ Usage
1. **Train the Model**
   Run training directly from the terminal:
   python model.train.py

This will:
Train multiple classifiers,
Select the best-performing model,
Save the trained model and label encoder,
Generate predictions (predictions.csv),
Save accuracy scores (model_performance.csv).

2. **Run the Flask App**
   python app.py
Go to 👉 http://localhost:5000
Use the JSON section for single predictions.
Use the CSV section to upload a file and download predictions.csv.

## 📊 Make Predictions

1. **JSON Input (Single Record)**
Paste input like this into the web form:
{
  "itching": 0,
  "high_fever": 1,
  "cough": 1,
  "chills": 1,
  "fatigue": 1,
  "weight_loss": 0
  .....
  ....
}
The output will display the predicted disease.

2. **CSV Upload (Multiple Records)**
Upload a .csv file with only symptom columns (no prognosis column).
The system will return a downloadable CSV with predictions appended.

## 📂 Project Structure
project_root/
│
├── app.py                          # Flask app (UI + endpoints)
├── model.train.py                  # Code to trigger the train.py script to run it from the terminal    
├── requirements.txt
├── README.md
│── gitignore
├── templates/
│   └── index.html                   # HTML UI
├── static/
│   └── style.css
│
├── data/
│   ├── raw/                         # Training & Testing CSVs
│ 
│
├── src/
│   ├── data/
│   │   └── preprocess.py            # Data loading & preprocessing
│   ├── models/
│   │   ├── train.py                 # Training script
│   │   └── predict.py               # Prediction functions
│   ├── utils/
│       ├── helpers.py               # Save/load functions
│       └── logger.py                # Logging setup
│
└── models/
    ├── trained_model.pkl
    └── label_encoder.pkl

## 🤖 Model Training Details
    Label Encoding: Target diseases converted via LabelEncoder.
    Preprocessing: StandardScaler → PCA (retain 95% variance).
    Classifiers Tested:
    Decision Tree
    Random Forest
    Gaussian Naive Bayes
    Linear SVM
    Logistic Regression
    The best-performing model (based on accuracy) is saved for predictions.

## 🛠️ Tech Stack
    | Component       | Tools / Libraries                          |
| --------------- | ------------------------------------------ |
| Backend         | Python, Flask                              |
| Modeling        | scikit-learn (pipelines, PCA, classifiers) |
| Data Processing | pandas, numpy                              |
| Logging         | Python logging module                      |
| CLI Training    | argparse                                   |
| UI              | HTML, CSS, minimal JS                      |

## 📜 License & Contact
© 2025 [PW Skills]
Licensed under the MIT License

📧 Contact:muzahidul.islam@pw.live

🌐 GitHub: @MuzahidulIslam1