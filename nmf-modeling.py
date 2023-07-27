import os
import re
from collections import Counter
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download stopwords from NLTK
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def remove_basic_words(text):
    # Define the set of basic words to remove
    basic_words = set(["the", "a", "an", "by", "more", "tim", "beiko", "nadia", "asparouhova", "2023", "reacted", "can", "what", "from", "as", "are", "s", "not", "t", "which", "have", "at", "like", "was", "be", "we", "or", "i", "they", "but", "you", "can't", "and", "in", "on", "of", "to", "for", "with", "that", "this", "is", "it"])

    # Use regex to split the text into words and filter out basic words
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in basic_words]

    return filtered_words

def remove_non_alphanumeric(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    basic_words_removed = [token for token in tokens if token not in stop_words]
    return remove_basic_words(' '.join(basic_words_removed))

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def analyze_text_files(directory):
    all_texts = []

    # Loop through each .txt file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()

                # Remove non-alphanumeric characters and extra whitespaces
                cleaned_text = remove_non_alphanumeric(text)

                # Tokenize, remove stopwords, and remove basic words
                tokens = tokenize_and_remove_stopwords(cleaned_text)

                # Lemmatize the tokens
                lemmatized_tokens = lemmatize_tokens(tokens)

                # Add the text to the list for topic modeling
                all_texts.append(' '.join(lemmatized_tokens))

    return all_texts

def print_topics(model, feature_names, num_words=10):
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-num_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        print(f"Topic {topic_idx}: {', '.join(top_features)}")

if __name__ == "__main__":
    # Replace "path/to/your/directory" with the actual path to the directory containing .txt files
    directory_path = "/Users/tim/code/sop-text/essays"

    all_texts = analyze_text_files(directory_path)

    # Vectorize the text using CountVectorizer
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    tf_matrix = vectorizer.fit_transform(all_texts)

    # Set the number of topics to extract
    num_topics = 5

    # Run NMF for topic modeling
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(tf_matrix)

    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Print the topics and their top words
    print_topics(nmf_model, feature_names)
