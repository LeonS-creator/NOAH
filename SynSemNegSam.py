import os
import pickle
import spacy
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def load_model_and_prepare_data(file_path, model_type='spacy', model_name="en_core_web_sm"):
    """
    Loads a text processing model and prepares textual data from a news file.
    This function can switch between SpaCy for syntactic processing and SentenceTransformers for semantic processing.

    Args:
        file_path (str): Path to the news file.
        model_type (str): Type of model to use ('spacy' or 'sentence_transformer').
        model_name (str): Name of the model to load.

    Returns:
        tuple: Dictionaries mapping news IDs to processed title and abstract vectors.
    """
    if model_type == 'spacy':
        model = spacy.load(model_name)
        vectorizer = lambda text: model(text).vector
    else:
        model = SentenceTransformer(model_name)
        vectorizer = lambda text: model.encode(text)

    nid_to_abstract, nid_to_title = {}, {}
    with tf.io.gfile.GFile(file_path, "r") as rd:
        for line in rd:
            nid, _, _, title, ab, _, _, _ = line.strip().split("\t")
            nid_to_abstract[nid] = vectorizer(ab) if ab else None
            nid_to_title[nid] = vectorizer(title) if title else None

    return nid_to_abstract, nid_to_title

def save_data(data, file_path):
    """
    Serializes data into a file using pickle.

    Args:
        data (dict): Data to serialize.
        file_path (str): File path where to save the serialized data.

    Returns:
        None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def calculate_cosine_similarity(pos_vector, neg_samples, vectors_dict):
    """
    Calculates cosine similarities between a positive vector and multiple negative sample vectors.

    Args:
        pos_vector (np.array): Vector representation of the positive sample.
        neg_samples (list): List of negative sample IDs.
        vectors_dict (dict): Dictionary of vector representations.

    Returns:
        dict: Sorted dictionary of negative samples by their similarity to the positive sample.
    """
    vectors = [vectors_dict[nid] for nid in neg_samples if nid in vectors_dict and vectors_dict[nid] is not None]
    if not vectors:
        return {}

    vectors = np.array(vectors)
    pos_vector = np.array(pos_vector).reshape(1, -1)
    similarities = cosine_similarity(pos_vector, vectors).flatten()
    return {nid: sim for nid, sim in zip(neg_samples, similarities)}

def process_behavior_data(nid_to_vector_path, behavior_file, output_path):
    """
    Processes user behavior to determine similarities between interacted news items and generates a model.

    Args:
        nid_to_vector_path (str): Path to the serialized dictionary of news ID to vector mappings.
        behavior_file (str): Path to the user behavior file.
        output_path (str): Path where the output data is saved.

    Returns:
        None
    """
    nid_to_vector = load_model_and_prepare_data(nid_to_vector_path)
    pos_neg_sim = {}

    with tf.io.gfile.GFile(behavior_file, "r") as rd:
        for line_index, line in enumerate(rd):
            _, _, _, impr = line.strip().split("\t")
            impr_news_labels = [i.split("-") for i in impr.split()]
            pos_samples = [nid for nid, label in impr_news_labels if label == '1']
            neg_samples = [nid for nid, label in impr_news_labels if label == '0']

            for pos in pos_samples:
                if pos in nid_to_vector:
                    pos_vector = nid_to_vector[pos]
                    pos_neg_sim[line_index] = calculate_cosine_similarity(pos_vector, neg_samples, nid_to_vector)

    save_data(pos_neg_sim, output_path)
    print(f"Processed {len(pos_neg_sim)} entries and saved to {output_path}")

# Example usage of syntactic and semantic processing
train_news_file = 'path/to/your/train_news_file.txt'
output_directory = "recommenders0/data"
nid_to_abstract, nid_to_title = load_model_and_prepare_data(train_news_file, model_type='spacy')

nid_to_abstract_path = "recommenders0/data/nid_to_abstract.pkl"
train_behaviors_file = "path/to/train_behaviors_file.txt"
output_file_path = "recommenders0/data/pos_neg_sim_abstract.pkl"
process_behavior_data(nid_to_abstract_path, train_behaviors_file, output_file_path)
