"""
this version of code utilizes multilingual-BERT model for the connection of german and english graphs nodes
"""

import pandas as pd
from gensim.models import Word2Vec
from collections import Counter
import re
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from gensim.models import FastText
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import XLMTokenizer, XLMWithLMHeadModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

f = open("nodes.txt", 'w')

olid_data = pd.read_table('olid.tsv')
olid_data = olid_data[olid_data['subtask_a'] == 'OFF']

germeval_data = pd.read_csv('germeval.txt', delimiter='\t', header=None, names=['tweet', 'tag1', 'tag2'])
germeval_data['text'] = germeval_data['tweet'].apply(lambda x: ' '.join(x.split()))
germeval_data['label'] = germeval_data['tag1'].apply(lambda x: x.split()[0])
germeval_data = germeval_data[germeval_data['label'] == 'OFFENSE']

german_data = germeval_data[['text', 'label']].values.tolist()
english_data = olid_data[['tweet', 'subtask_a']].values.tolist()

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
stop_words_en = set(stopwords.words('english'))  # Set of English stop words
stop_words_de = set(stopwords.words('german'))  # Set of German stop words


nlp_en = spacy.load('en_core_web_sm')
nlp_de = spacy.load('de_core_news_sm')

print("Loading english word embeddings")
english_muse_embeddings = KeyedVectors.load_word2vec_format('english_word_embedding.txt')
print("Loading german word embeddings")
german_muse_embeddings = KeyedVectors.load_word2vec_format('german_word_embedding.txt')

def preprocess_text(text, language):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove mentioned users
    text = re.sub(r"@\w+", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9ÄäÖöÜüßẞ#]", " ", text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    if language == 'english':
        tokens = [token for token in tokens if token not in stop_words_en]
        lemmas = [token.lemma_ for token in nlp_en(' '.join(tokens))]
        tokens = lemmas
    elif language == 'german':
        tokens = [token for token in tokens if token not in stop_words_de]
        lemmas = [token.lemma_ for token in nlp_de(' '.join(tokens))]
        tokens = lemmas
    # Join tokens back into a preprocessed text
    preprocessed_text = ' '.join(tokens)
    # Merge hashtag and word together
    preprocessed_text = re.sub(r"#\s*(\w+)", r"#\1", preprocessed_text)
    return preprocessed_text

print("Processing english data")
english_data_preprocessed = [(preprocess_text(text, 'english'), label) for text, label in english_data]
print("Processing german data")
german_data_preprocessed = [(preprocess_text(text, 'german'), label) for text, label in german_data]

english_sentences = [text.split() for text, _ in english_data_preprocessed]
german_sentences = [text.split() for text, _ in german_data_preprocessed]

print("Creating English model")
english_model = Word2Vec(sentences=english_sentences, vector_size=100, min_count=1)
print("Creating German model")
german_model = Word2Vec(sentences=german_sentences, vector_size=100, min_count=1)

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
model = AutoModel.from_pretrained('bert-base-multilingual-cased')

def calculate_similarity_multilingual(word1, word2):
    # Tokenize the words
    inputs = tokenizer([word1, word2], return_tensors='pt', padding=True, truncation=True)
    # Obtain the token embeddings from mBERT
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    # Calculate the cosine similarity between the token embeddings
    similarity = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    return similarity.item()

def get_word_embedding(word, model, lang):
    if lang == 'english':
        if word in english_muse_embeddings:
            return english_muse_embeddings[word]
    
    else:
        if word in german_muse_embeddings:
            return german_muse_embeddings[word]
    
    if word in model.wv:
        return model.wv[word]
    return None

def calculate_similarity(word1, word2, model1, model2, lang1, lang2):
    embedding1 = get_word_embedding(word1, model1, lang1)
    embedding2 = get_word_embedding(word2, model2, lang2)
    if embedding1 is not None and embedding2 is not None:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return 0

def build_graph(data, model, language):
    graph = nx.Graph()
    word_count = Counter()

    # Build word co-occurrence count
    for text, label in data:
        words = text.split()
        entities = []  # List to store identified entities in the text
        entities.append(text)       
        if language == 'english':
            doc = nlp_en(text)
        elif language == 'german':
            doc = nlp_de(text)
        hashtag_seen = False
        for token in doc:
            if token.text.startswith('#'):
                hashtag_seen = True
                continue
            if (len(token.text) < 3 or token.text == "--"):
                hashtag_seen = False
                continue
            if(hashtag_seen):
                entity = token.text + "_HASHTAG"
                entities.append(entity)
                continue
            entity = token.lemma_
            if token.ent_type_:
                entity += '_' + token.ent_type_
            entities.append(entity)
        word_count.update(entities)
        for entity in entities:
            if not graph.has_node(entity):
                graph.add_node(entity)
            for other_entity in entities:
                if entity != other_entity:
                    if not graph.has_edge(entity, other_entity):
                        graph.add_edge(entity, other_entity, weight=0)
                    graph[entity][other_entity]['weight'] += 1

                    # Word Sense Disambiguation (using Word2Vec similarity)
                    similarity = calculate_similarity(entity.split('_')[0], other_entity.split('_')[0], model, model, language, language)
                    graph.add_edge(entity, other_entity, weight=similarity)

    # Normalize edge weights using TF-IDF formula
    total_documents = len(data)
    for word1, word2, data in graph.edges(data=True):
        weight = data['weight']
        tf = weight / word_count[word1]
        idf = np.log(total_documents / word_count[word1])
        tf_idf = tf * idf
        graph[word1][word2]['weight'] = tf_idf

    return graph

print("Creating english graph")
english_graph = build_graph(english_data_preprocessed, english_model, 'english')
print(len(english_graph.nodes))
print(english_graph.number_of_edges())

print("Creating german graph")
german_graph = build_graph(german_data_preprocessed, german_model, 'german')
print(len(german_graph.nodes))
print(german_graph.number_of_edges())

graph = nx.Graph()
graph.add_nodes_from(english_graph.nodes(data=True))
graph.add_edges_from(english_graph.edges(data=True))
graph.add_nodes_from(german_graph.nodes(data=True))
graph.add_edges_from(german_graph.edges(data=True))

print(graph.number_of_edges())

# Connect nodes between English and German graphs using knowledge graph concepts
for english_node, english_data in english_graph.nodes(data=True):
    for german_node, german_data in german_graph.nodes(data=True):
        english_node_parts = english_node.split('_')
        german_node_parts = german_node.split('_')
        
        # Check if both English and German nodes have types
        if len(english_node_parts) == 2 and len(german_node_parts) == 2:
            
            eng_text, english_node_type = english_node_parts
            ger_text, german_node_type = german_node_parts
            
            # Check if the types of the English and German nodes match
            if english_node_type == german_node_type:
                
                if(english_node_type == "HASHTAG"):
                    continue
                similarity = calculate_similarity_multilingual(eng_text, ger_text)
                print(similarity)
                # Adjust the threshold based on your requirements
                if similarity > 0.75:
                    print(english_node + " <-> " + german_node, similarity, file=f)
                    graph.add_edge(english_node, german_node, weight=similarity)

        elif len(english_node_parts) == 1 and len(german_node_parts) == 1:
            
            if(len(english_node) > 15 or len(german_node) > 15):
                continue
            
            similarity = calculate_similarity_multilingual(english_node, german_node)
            print(similarity)
            if similarity > 0.75:
                print(english_node + " <-> " + german_node, similarity, file=f)
                graph.add_edge(english_node, german_node, weight=similarity)

        else:
            eng_text = english_node_parts[0]
            ger_text = german_node_parts[0]

            if(len(english_node) > 15 or len(german_node) > 15):
                continue
            
            similarity = calculate_similarity_multilingual(eng_text, ger_text)
            print(similarity)
            if similarity > 0.75:
                print(english_node + " <-> " + german_node, similarity, file=f)
                graph.add_edge(english_node, german_node, weight=similarity)

print(graph.number_of_edges())
