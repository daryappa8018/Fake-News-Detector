# Fake News Detection using Machine Learning

## Overview
This project implements a machine learning pipeline to classify news articles as **Fake News** or **Real News** based on their textual content. The system uses multiple classifiers trained on a labeled dataset and applies TF-IDF vectorization for feature extraction from the text.

---

## Features
- Loads and processes two datasets: fake news and real news.
- Cleans and preprocesses the text data by removing URLs, punctuation, HTML tags, and other noise.
- Vectorizes text data using TF-IDF (Term Frequency-Inverse Document Frequency).
- Trains four different classifiers:
  - Logistic Regression
  - Decision Tree
  - Gradient Boosting Classifier
  - Random Forest Classifier
- Evaluates the models on a test set and prints detailed classification reports.
- Provides a simple interface to manually test news articles by inputting text and getting predictions from all models.

---

## Dataset
- **Fake.csv**: Contains labeled fake news articles.
- **True.csv**: Contains labeled real news articles.

*Both datasets should be placed in the project directory.*

---

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn

Install dependencies using pip:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## How to Use

1. Place `Fake.csv` and `True.csv` in the project folder.
2. Run the Python script:
   ```bash
   python fake_news_detection.py
   ```
3. The script will train the models, display classification reports for each, and then prompt you to enter any news text.
4. Input a news article text to get predictions from all models on whether it is fake or real.

---

## Code Structure

- Data loading and labeling
- Data cleaning and preprocessing
- TF-IDF vectorization
- Model training and evaluation
- Manual testing function to predict new samples

---

## Future Work

- Improve model accuracy with more advanced NLP techniques like word embeddings.
- Use deep learning models such as LSTM or Transformers.
- Develop a web interface for real-time fake news detection.
- Extend datasets for better generalization.

---

## Author

**Daryappa Mane**, 
**Ishant Nandeshwar** 
Student, Computer Science Engineering  
COEP Technological University, Pune


