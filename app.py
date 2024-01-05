import streamlit as st
import pickle
import re
import nltk
import PyPDF2
import io

nltk.download('stopwords')

# Loading models
model = pickle.load(open('model_with_lr.pkl', 'rb'))
tfidf = pickle.load(open('tfidf (1).pkl', 'rb'))

def extract_text_from_pdf(pdf_bytes):
    pdf_text = ""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    for page_num in range(len(pdf_reader.pages)):
        pdf_text += pdf_reader.pages[page_num].extract_text()
    return pdf_text

def clean_resume(txt):
    clean_txt = re.sub('http\S+\s', ' ', txt)
    clean_txt = re.sub('@\S+', ' ', clean_txt)
    clean_txt = re.sub('#\S+', ' ', clean_txt)
    clean_txt = re.sub('RT|cc', ' ', clean_txt)
    clean_txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_{|}~"""), ' ', clean_txt)
    clean_txt = re.sub(r'[^\x00-\x7f]', ' ', clean_txt)
    clean_txt = re.sub('\s+', ' ', clean_txt)
    return clean_txt

# web app
def main():
    st.title("Resume Screening For Job Description")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            # Extract text from PDF
            resume_text = extract_text_from_pdf(uploaded_file.read())

            # Clean the resume text
            cleaned_resume = clean_resume(resume_text)

            # Perform the prediction
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = model.predict(input_features)[0]

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")

            # Display the results
            st.write("Predicted Category:", category_name)

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
