import streamlit as st

st.markdown("# Predicting Product Review Sentiment Using Classification")

st.markdown("""The goal of this project is to build a classification machine learning (ML) pipeline in a web application to use as a tool to analyze the models and gain useful insights about model performance. Using trained classification models, build a ML application that predicts whether a product review is positive or negative.

- Build end-to-end classification pipeline with four classifiers 1) Logistic Regression, 2) Stochastic Gradient Descent, 3) Stochastic Gradient Descent with Cross Validation, and 4) Majority Class.
- Evaluate classification methods using standard metrics including precision, recall, and accuracy, ROC Curves, and area under the curve.
- Develop a web application that walks users through steps of the classification pipeline and provide tools to analyze multiple methods across multiple metrics. 
- Develop a web application that classifies products as positive or negative and indicates the cost of displaying false positives and false negatives using a specified model.
""")

st.markdown(""" Amazon Products Dataset

Millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

We have added additional features to the dataset. There are many features, but the important ones include:
- name: name of Amazon product	
- reviews.text: text in review	
- reviews.title: title of reviews	
""")

st.markdown("Click **Explore Dataset** to get started.")