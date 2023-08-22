import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from helper_functions import fetch_dataset, clean_data, summarize_review_data, display_review_keyword
import string
import re
from langdetect import detect
import numpy as np
# import matplotlib as plt
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.cm as cm
from wordcloud import WordCloud

#############################################

st.markdown("# Predicting Product Review Sentiment Using Classification")

st.markdown('# Explore & Preprocess Dataset')

#############################################

def plot_ngrams(df, n=2):
    """
    This function creates and displays n-gram frequency plots
    """
    words = word_tokenize(" ".join(df["reviews"].tolist()))
    n_grams = list(ngrams(words, n))
    n_grams_freq = Counter(n_grams)

    n_grams_df = pd.DataFrame.from_dict(n_grams_freq, orient='index').reset_index()
    n_grams_df = n_grams_df.sort_values(by=[0], ascending=False)
    n_grams_df.columns = ['N-gram', 'Frequency']

    colors = np.linspace(0, 0.7, 20)
    custom_cmap = cm.get_cmap('Blues_r')(colors)
    plt.figure(figsize=(10, 5))
    plt.bar(n_grams_df['N-gram'].apply(lambda x: ' '.join(x)).values[:20], n_grams_df['Frequency'].values[:20], color=custom_cmap)
    plt.title(f'{n}-gram Frequency Plot', fontsize=14, fontweight='bold')
    plt.xlabel('N-grams', fontsize=12, fontstyle='italic')
    plt.ylabel('Frequency', fontsize=12, fontstyle='italic')
    plt.xticks(rotation=90)
    plt.grid(True, axis='y')
    st.pyplot(plt.gcf())

def create_time_series_plots(df, feature, date_col='date'): #requires time - do we have this? 
    """
    This function creates time series plots
    """
    df[date_col] = pd.to_datetime(df[date_col])
    time_series_df = df.set_index(date_col)
    time_series_df[feature].resample('D').mean().plot()
    st.pyplot(plt.gcf())

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def remove_punctuation(df, features):
    """
    This function removes punctuation from features (i.e., product reviews)

    Input: 
        - df: the pandas dataframe
        - feature: the features to remove punctation
    Output: 
        - df: dataframe with updated feature with removed punctuation
    """
    translator = str.maketrans('', '', string.punctuation)
    for feature_name in features:
        # check if the feature contains string or not
        if df[feature_name].dtype == 'object':
            # applying translate method eliminating punctuations
            df[feature_name] = df[feature_name].apply(
                lambda x: x.translate(translator) if isinstance(x, str) else x)

    # Store new features in st.session_state
    st.session_state['data'] = df

    # Confirmation statement
    st.write('Punctuation was removed from {}'.format(features))
    return df

def transform_lowercase(df, features):
    """
    This function transforms uppercase to lowercase in the specified features (i.e., product reviews).

    Input: 
        - df: the pandas DataFrame
        - features: the features to transform lowercase
    Output: 
        - df: DataFrame with updated features transformed to lowercase
    """
    for feature_name in features:
        # check if the feature contains string or not
        if df[feature_name].dtype == 'object':
            # applying lower method to transform to lowercase
            df[feature_name] = df[feature_name].apply(
                lambda x: x.lower() if isinstance(x, str) else x)

    # Store new features in st.session_state
    st.session_state['data'] = df

    # Confirmation statement
    st.write('Uppercase was transformed to lowercase in {}'.format(features))
    return df

def remove_special_chars(df, features):
    """
    This function removes URLs, special characters, and numbers from the specified features (reviews).

    Input: 
        - df: the pandas DataFrame
        - features: the features to remove special characters
    Output: 
        - df: DataFrame with updated features without special characters
    """
    for feature_name in features:
        # check if the feature contains string or not
        if df[feature_name].dtype == 'object':
            # removing URLs
            df[feature_name] = df[feature_name].apply(
                lambda x: re.sub('http[^\s]+', '', x) if isinstance(x, str) else x)
            
            # removing special characters and numbers
            df[feature_name] = df[feature_name].apply(
                lambda x: re.sub('[^a-zA-Z\s]', '', x) if isinstance(x, str) else x)
    
    # Store new features in st.session_state
    st.session_state['data'] = df
    
    # Confirmation statement
    st.write('URLs, special characters, and numbers were removed from {}'.format(features))
    return df

def keep_english_reviews(df, features):
    """
    This function keeps only the reviews in English from the specified features.

    Input: 
        - df: the pandas DataFrame
        - features: the features to filter English reviews
    Output: 
        - df: DataFrame with only English reviews in the specified features
    """
    for feature_name in features:
        # check if the feature contains string or not
        if df[feature_name].dtype == 'object':
            df[feature_name] = df[feature_name].apply(
                lambda x: x if isinstance(x, str) and x.strip() != '' and detect(x) == 'en' else np.nan)

    # Drop rows with NaN values in the specified features
    df.dropna(subset=features, inplace=True)
    
    # Confirmation statement
    st.write('Kept only English reviews in {}'.format(features))
    return df

def remove_stopwords(df, features):
    """
    This function removes stopwords from the specified features (reviews).

    Input: 
        - df: the pandas DataFrame
        - features: the features to remove stopwords
    Output: 
        - df: DataFrame with stopwords removed from the specified features
    """
    stop_words = set(stopwords.words('english'))

    for feature_name in features:
        # check if the feature contains string or not
        if df[feature_name].dtype == 'object':
            df[feature_name] = df[feature_name].apply(
                lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]) if isinstance(x, str) else x)

    st.session_state['data'] = df
    st.write('Stopwords were removed from {}'.format(features))
    return df

def word_count_encoder(df, feature, word_encoder):
    """
    This function performs word count encoding on feature in the dataframe

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform word count encoding
        - word_encoder: list of strings with word encoding names 'TF-IDF', 'Word Count'
    Output: 
        - df: dataframe with word count feature
    """
    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform(df[feature])
    word_count_df = pd.DataFrame(X_train_counts.toarray())
    word_count_df = word_count_df.add_prefix('word_count_')
    df = pd.concat([df, word_count_df], axis=1)

    # Show confirmation statement
    st.write('Feature {} has been word count encoded from {} reviews.'.format(
        feature, len(word_count_df)))

    # Store new features in st.session_state
    st.session_state['data'] = df

    word_encoder.append('Word Count')
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = count_vect

    return df

def tf_idf_encoder(df, feature, word_encoder):
    """
    This function performs tf-idf encoding on the given features

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform tf-idf encoding
        - word_encoder: list of strings with word encoding names 'TF-IDF', 'Word Count'
    Output: 
        - df: dataframe with tf-idf encoded feature
    """
    tfidf_transformer = TfidfTransformer()
    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform(df[feature])
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    word_count_df = pd.DataFrame(X_train_tfidf.toarray())
    word_count_df = word_count_df.add_prefix('tf_idf_word_count_')
    df = pd.concat([df, word_count_df], axis=1)

    # Show confirmation statement
    st.write(
        'Feature {} has been TF-IDF encoded from {} reviews.'.format(feature, len(word_count_df)))

    # Store new features in st.session_state
    st.session_state['data'] = df

    word_encoder.append('TF-IDF')
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = count_vect
    st.session_state['tfidf_transformer'] = tfidf_transformer
    return df

def impute_dataset(X, impute_method):
    """
    Replace missing values with mode, min, max

    Input: dataframe, method to impute
    Output: Returns imputed Dataframe
    """
    df = X.copy()
    if impute_method == 'Min':
        df.fillna(1.0, inplace = True)
    elif impute_method == 'Max':
        df.fillna(5.0, inplace = True)
    elif impute_method == 'Mode':
        df.fillna(X.mode(), inplace = True)
    return df

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:
    df_og = df
    # Display original dataframe
    st.markdown('You have uploaded the Amazon Product Reviews dataset.')
    st.markdown('Raw data:')
    st.dataframe(df_og)
    st.markdown('The dataframe consists of {} rows/reviews, with {} columns each.'.format(df.shape[0], df.shape[1]))

    # Remove irrelevant features
    df, data_cleaned = clean_data(df)
    if (data_cleaned):
        st.markdown('New dataframe where only the following features are kept: text, title, rating.')
        st.dataframe(df)
    
    st.markdown('### Clean dataset')

    # Remove Punctation
    st.markdown('##### Remove punctuation from features')
    removed_p_features = st.multiselect(
        'Select features to remove punctuation',
        df.columns,
    )
    if (removed_p_features):
        df = remove_punctuation(df, removed_p_features)

    # Transform to lower case
    st.markdown('##### Transform text to lower case')
    lowerc = st.multiselect(
        'Select features to alter',
        df.columns,
    )
    if (lowerc):
        df = transform_lowercase(df, lowerc)
        
    # remove special characters, urls and numbers
    st.markdown('##### Remove special characters, URLs and numbers')
    spec_char = st.multiselect(
        'Select features to remove special characters',
        df.columns,
    )
    if (spec_char):
        df = remove_special_chars(df, spec_char)

    # remove special characters, urls and numbers
    st.markdown('##### Remove all reviews that are not in English')
    english_only = st.multiselect(
        'Filter reviews in English only',
        df.columns,
    )
    if (english_only):
        df = keep_english_reviews(df, english_only)

    # remove stopwords
    st.markdown('##### Remove stopwords')
    stop_w = st.multiselect(
        'Select features to remove stopwords from',
        df.columns,
    )
    if (spec_char):
        df = remove_stopwords(df, stop_w)

    st.markdown('### Handle Missing Values')
    st.markdown('- View initial data with missing values or invalid inputs')
    # Show summary of missing values including the 1) number of categories with missing values, average number of missing values per category, and Total number of missing values
    st.markdown('   Number of categories with missing values: {0:.2f}'.format(df.isna().any(axis=0).sum()))
    for cat in list(df.columns):
        val = df[cat].isna().any(axis=0).sum()
        st.markdown(f'  Number of missing values for category {cat}: {df[str(cat)].isna().sum()}')
    st.markdown(f'  Total number of missing values: {df.isna().sum().sum()}')

    st.markdown('- Replace numerical missing values to 0, mean, or median')

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
    select_impute_ft = st.multiselect(
            'Select a feature to impute',
            options=numeric_columns
    )
    select_impute_opt = st.selectbox(
            'Select a method to impute data',
            options=['Mode', 'Min', 'Max']
    )
    # Call impute_dataset function to resolve data handling/cleaning problems by calling impute_dataset
    if (select_impute_ft and select_impute_opt):
        try:
            df_imp = impute_dataset(df[select_impute_ft], select_impute_opt)
            st.write(df_imp) 
        except Exception as e:
            print(e)

    st.markdown('### Dataset Review')
    # See dataset
    if st.button('Reveal final dataset'):
        st.dataframe(df)

    # Summarize reviews
    st.markdown('##### Reviews Summary')
    object_columns = df.select_dtypes(include=['object']).columns
    summarize_reviews = st.selectbox(
        'Select the reviews from the dataset',
        object_columns,
    )
    if(summarize_reviews):
        # Show summary of reviews
        summary = summarize_review_data(df, summarize_reviews)

    # Inspect Reviews
    st.markdown('##### Inspect Reviews')

    review_keyword = st.text_input(
        "Enter a keyword to search in reviews",
        key="review_keyword",
    )
    if (review_keyword):
        displaying_review = display_review_keyword(df, review_keyword)
        st.write(displaying_review)


    st.markdown('### Data Visualizations')
    if st.button('Explore Data') or 'explore_data_pressed' in st.session_state:
        st.session_state.explore_data_pressed = True

        rev = df['reviews']
        text = ' '.join(rev.dropna().tolist())
        st.markdown('#### Word Cloud')
        generate_word_cloud(text)

        prep_df = st.session_state['data']
        prep_df = prep_df.drop(['reviews'], axis=1)
        prep_df = st.session_state['data']
        
        prep_df = st.session_state['data']
        st.markdown("#### Word Frequency Plot")
        words = word_tokenize(" ".join(prep_df["reviews"].tolist()))
        word_freq = Counter(words)
        word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
        word_freq_df = word_freq_df.sort_values(by=[0], ascending=False)
        word_freq_df.columns = ['Word', 'Frequency']
        st.write('Top 20 most frequent words')
        # plot
        colors = np.linspace(0, 1, 20)
        custom_cmap = cm.get_cmap('rainbow')(colors)
        plt.figure(figsize=(10, 5))
        plt.bar(word_freq_df['Word'].values[:20], word_freq_df['Frequency'].values[:20], color=custom_cmap)
        plt.grid(True, axis='y')
        plt.xlabel('Words', fontsize=12, fontstyle='italic')
        plt.ylabel('Frequency', fontsize=12, fontstyle='italic')
        plt.title('Word Frequency Plot', fontsize=14, fontweight='bold')
        plt.xticks(rotation=40)
        st.pyplot(plt.gcf())
        # table to display top 20 frequent words in dataframe
        st.write(word_freq_df.head(20))  

        st.markdown("#### N-gram Frequency Plots")
        plot_ngrams(prep_df, n=2)  # bigrams
        n = st.slider("Select the value of n for n-grams", min_value=3, max_value=6, step=1)
        if n:
            plot_ngrams(prep_df, n)


    # Handling Text and Categorical Attributes
    st.markdown('### Handling Text Attributes')
    string_columns = list(df.select_dtypes(['object']).columns)
    word_encoder = []

    word_count_col, tf_idf_col = st.columns(2)

    # Perform Word Count Encoding
    with (word_count_col):
        text_feature_select_int = st.selectbox(
            'Select text features for encoding word count',
            string_columns,
        )
        if (text_feature_select_int and st.button('Word Count Encoder')):
            df = word_count_encoder(df, text_feature_select_int, word_encoder)

    # Perform TF-IDF Encoding
    with (tf_idf_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for encoding TF-IDF',
            string_columns,
        )
        if (text_feature_select_onehot and st.button('TF-IDF Encoder')):
            df = tf_idf_encoder(df, text_feature_select_onehot, word_encoder)

    # Show updated dataset
    if (text_feature_select_int or text_feature_select_onehot):
        st.write(df)

    # Save dataset in session_state
    st.session_state['data'] = df

    st.write('Continue to Train Model')
