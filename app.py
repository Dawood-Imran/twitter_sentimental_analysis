import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import streamlit as st
import pandas as pd


import nltk
nltk.download('stopwords')

port_stem = PorterStemmer()


with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)


def stemming(content):
  
  # removing all the things which are not alphabets
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  # converting to lower case
  stemmed_content = stemmed_content.lower()
  # tokenizing the content
  stemmed_content = stemmed_content.split()
  # Stemming
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  # joining them on spaces
  stemmed_content = ' '.join(stemmed_content)

  return stemmed_content


  

st.title("Twitter Sentiment Analysis")

# Description
st.write("""
    This app uses machine learning to predict the sentiment of a given tweet.
    Enter a tweet below and find out whether it is positive, or negative!
""")

# Input tweet text
input_tweet = st.text_area("Enter a tweet:", "")

# Process and predict the sentiment if the input is not empty
if st.button("Analyze Sentiment"):
    if input_tweet:
        # Transform the input using the loaded TF-IDF Vectorizer
        input_data = loaded_vectorizer.transform([input_tweet])
        
        # Make prediction
        prediction = loaded_model.predict(input_data)[0]

        # Get the prediction probabilities for more detail
        probabilities = loaded_model.predict_proba(input_data)

        # Define sentiment labels
        sentiment_map = {0: 'Negative', 1: 'Positive'}

        # Display the sentiment
        st.write(f"Predicted Sentiment: **{sentiment_map[prediction]}**")

        # Display prediction probabilities
        st.write(f"Confidence: {probabilities[0][prediction] * 100:.2f}%")

        # Display the probabilities for all classes
        st.write("Probability Distribution:")
        st.write(pd.DataFrame(probabilities, columns=['Negative', 'Positive']).T)
    else:
        st.warning("Please enter a tweet to analyze.")

# Footer
st.write("Made with ❤️ by Dawood Imran")
  



  





