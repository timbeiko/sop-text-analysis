import os
import re
from collections import Counter
import math
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora

def remove_basic_words(text):
    # Define the set of basic words to remove
    basic_words = set(["the", "a", "an", "by", "more", "tim", "beiko", "nadia", "asparouhova", "2023", "reacted", "can", "what", "from", "as", "are", "s", "not", "t", "which", "have", "at", "like", "was", "be", "we", "or", "i", "they", "but", "you", "can't", "and", "in", "on", "of", "to", "for", "with", "that", "this", "is", "it"])

    # Use regex to split the text into words and filter out basic words
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in basic_words]

    return filtered_words

def analyze_text_files(directory):
    word_occurrences = Counter()
    pair_occurrences = Counter()
    triple_occurrences = Counter()
    file_word_count = Counter()
    all_texts = []

    # Loop through each .txt file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()

                # Remove basic words from the text
                words = remove_basic_words(text)

                # Update word occurrences
                word_occurrences.update(words)

                # Update pair occurrences
                pairs = extract_ngrams(words, 2)
                pair_occurrences.update(pairs)

                # Update triple occurrences
                triples = extract_ngrams(words, 3)
                triple_occurrences.update(triples)

                # Update file_word_count with unique words appearing in the current file
                file_word_count.update(set(words))

                # Add the text to the list for topic modeling
                all_texts.append(words)

    return word_occurrences, pair_occurrences, triple_occurrences, file_word_count, all_texts

def extract_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]

def calculate_tf_idf(tf, idf):
    return tf * idf

def calculate_idf(total_files, document_frequency):
    return math.log(total_files / (1 + document_frequency))

def topic_modeling(texts, num_topics=5, num_words=10):
    # Create a dictionary representation of the essays
    dictionary = corpora.Dictionary(texts)

    # Create a corpus from the dictionary
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Print the topics and their top words
    for idx, topic in lda_model.print_topics(-1, num_words=num_words):
        print(f"Topic {idx}: {topic}")

def print_sorted_by_tf_idf(tf_idf_scores, title, n=10):
    print(f"Sorted {title} based on tf-idf:")
    sorted_items = sorted(tf_idf_scores.items(), key=lambda item: item[1], reverse=True)
    for word, score in sorted_items[:n]:
        print(f"Word: {word}, TF-IDF Score: {score:.6f}")

if __name__ == "__main__":
    # Replace "path/to/your/directory" with the actual path to the directory containing .txt files
    directory_path = "/Users/tim/code/sop-text/essays"

    word_occurrences, pair_occurrences, triple_occurrences, file_word_count, all_texts = analyze_text_files(directory_path)

    total_files = len([filename for filename in os.listdir(directory_path) if filename.endswith(".txt")])

    tf_idf_word_scores = {}
    for word, tf in word_occurrences.items():
        idf = calculate_idf(total_files, file_word_count[word])
        tf_idf_word_scores[word] = calculate_tf_idf(tf, idf)

    tf_idf_pair_scores = {}
    for pair, tf in pair_occurrences.items():
        idf = calculate_idf(total_files, file_word_count[pair[0]])
        tf_idf_pair_scores[pair] = calculate_tf_idf(tf, idf)

    tf_idf_triple_scores = {}
    for triple, tf in triple_occurrences.items():
        idf = calculate_idf(total_files, file_word_count[triple[0]])
        tf_idf_triple_scores[triple] = calculate_tf_idf(tf, idf)

    print_sorted_by_tf_idf(tf_idf_word_scores, "words")
    print_sorted_by_tf_idf(tf_idf_pair_scores, "word pairs")
    print_sorted_by_tf_idf(tf_idf_triple_scores, "word triples")

    # Topic Modeling using LDA
    num_topics = 5  # Set the number of topics to extract
    num_words_per_topic = 10  # Number of words to show for each topic
    topic_modeling(all_texts, num_topics=num_topics, num_words=num_words_per_topic)
