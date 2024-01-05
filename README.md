# Resume_Screening_for_Job_description
This project implements a machine learning model that analyzes resumes and automatically classifies them based on the corresponding job descriptions. It aims to help recruiters, HR professionals, and hiring managers efficiently screen large volumes of resumes by identifying the most relevant candidates for specific roles.

## Data Exploration:

The dataset which was downloaded from kaggle contains 2544 resumes across 25 unique job categories.
The most common categories are Data Science, HR, and Software Engineering.

Text Preprocessing:
Resume text is cleaned to remove noise and irrelevant information, including:
URLs
Mentions (@usernames)
Hashtags
RT and cc tags
Punctuation and special characters
Non-ASCII characters
Extra whitespace

Feature Engineering:
TF-IDF vectorization is used to represent the text data as numerical feature vectors, emphasizing important words and their frequencies.

## Modeling
The project explores various classification models for resume screening:
Logistic Regression
KNeighbors Classifier
Random Forest
Multinomial Naive Bayes

Performance Results:

Accuracy: 99.48% 
Precision : 99.5%
Recall : 99.68%
F1-Score : 99.34%

## Save
Use the pickle library to save & load the model

## Usage:

Install Dependencies:

Ensure you have the following libraries installed:
Streamlit
Pickle
NLTK
PyPDF2
NumPy
Pandas
Scikit-learn
You can install them using pip: pip install streamlit pickle nltk PyPDF2 numpy pandas sklearn
Run the Web App:

Navigate to the project directory in your terminal.
Execute the command: streamlit run app.py 
(replace app.py with the actual filename if different).

Upload a Resume:

In the web app interface, click the "Upload Resume" button.

Select a resume file in either TXT or PDF format.

View Prediction:
The app will process the resume and display the predicted job category.
