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
import json

def preprocess_text(text, language):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove mentioned users
    text = re.sub(r"@\w+", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-ZÄäÖöÜüßẞ#]", " ", text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Join tokens back into a preprocessed text
    preprocessed_text = ' '.join(tokens)
    # Merge hashtag and word together
    preprocessed_text = re.sub(r"#\s*(\w+)", r"#\1", preprocessed_text)
    return preprocessed_text

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
            if(token.text in stop_words_en or token.text in stop_words_de):
                continue
            if token.text.startswith('#'):
                hashtag_seen = True
                continue
            if (len(token.text) < 3):
                hashtag_seen = False
                continue
            if(hashtag_seen):
                entity = token.text + "_HASHTAG"
                entities.append(entity)
                continue
            entity = token.lemma_
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

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words_en = set(stopwords.words('english'))  # Set of English stop words
stop_words_de = set(stopwords.words('german'))  # Set of German stop words
stop_words = stop_words_en.union(stop_words_de)

nlp_en = spacy.load('en_core_web_sm')
nlp_de = spacy.load('de_core_news_sm')


nodes_file = open("nodes.txt", 'w')
dict_file = open("data/knowledge.json", "r")

en_dict = json.load(dict_file)

olid_data = pd.read_table('data/olid.tsv')
olid_data = olid_data[olid_data['subtask_a'] == 'OFF']
english_data = olid_data[['tweet', 'subtask_a']].values.tolist()

germeval_data = pd.read_csv('data/germeval.txt', delimiter='\t', header=None, names=['tweet', 'tag1', 'tag2'])
germeval_data['text'] = germeval_data['tweet'].apply(lambda x: ' '.join(x.split()))
germeval_data['label'] = germeval_data['tag1'].apply(lambda x: x.split()[0])
germeval_data = germeval_data[germeval_data['label'] == 'OFFENSE']
german_data = germeval_data[['text', 'label']].values.tolist()

profanity = pd.read_csv("data/profanity_en.csv")
profanity = profanity['text'].values.tolist()

print("Loading english word embeddings")
english_muse_embeddings = KeyedVectors.load_word2vec_format('data/english_word_embedding.txt')
print("Processing english data")
english_data_preprocessed = [(preprocess_text(text, 'english'), label) for text, label in english_data]
english_sentences = [text.split() for text, _ in english_data_preprocessed]

print("Loading german word embeddings")
german_muse_embeddings = KeyedVectors.load_word2vec_format('data/german_word_embedding.txt')
print("Processing german data")
german_data_preprocessed = [(preprocess_text(text, 'german'), label) for text, label in german_data]
german_sentences = [text.split() for text, _ in german_data_preprocessed]

print("Creating English model")
english_model = Word2Vec(sentences=english_sentences, vector_size=100, min_count=1)
print("Creating German model")
german_model = Word2Vec(sentences=german_sentences, vector_size=100, min_count=1)

#print("Creating BERT model and tokenizer")
#tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
#model = AutoModel.from_pretrained('bert-base-multilingual-cased')

en_layer = open("offensive_dataset.adj1", "w")
de_layer = open("offensive_dataset.adj2", "w")

print("Creating english graph")
english_graph = build_graph(english_data_preprocessed, english_model, 'english')
print("English graph nodes:", len(english_graph.nodes))
print("English graph edges:", english_graph.number_of_edges())

i = 0
en_nodes = {}
en_neighbours = {}
for n in english_graph.nodes:
    if n not in en_nodes:
        en_nodes[n] = i
        i += 1

for n in english_graph.nodes:
    connections = list(english_graph.neighbors(n))
    for c in connections:
        print(en_nodes[n], en_nodes[c], file = en_layer)

print("Creating german graph")
german_graph = build_graph(german_data_preprocessed, german_model, 'german')
with open("xx.txt", "w") as d:
    print(german_graph.nodes, file=d)
print("German graph nodes:", len(german_graph.nodes))
print("German graph edges:", german_graph.number_of_edges())

i = 0
de_nodes = {}
de_neighbours = {}
for n in german_graph.nodes:
    if n not in de_nodes:
        de_nodes[n] = i
        i += 1

for n in german_graph.nodes:
    connections = list(german_graph.neighbors(n))
    for c in connections:
        print(de_nodes[n], de_nodes[c], file = de_layer)


print("Merging english and german graph")
graph = nx.Graph()
graph.add_nodes_from(english_graph.nodes(data=True))
graph.add_edges_from(english_graph.edges(data=True))
graph.add_nodes_from(german_graph.nodes(data=True))
graph.add_edges_from(german_graph.edges(data=True))

print(graph.number_of_edges())
eng_nodes = english_graph.nodes
deu_nodes = german_graph.nodes

for key, values in en_dict.items():
    for node in eng_nodes:
        if key == node or (" " + key + " ") in node:
            for value in values:
                for de in deu_nodes:
                    if (value.replace("-", " ").replace("_", " ")) == de or (" " + (value.replace("-", " ").replace("_", " ")) + " ") in de:
                        graph.add_edge(node, de, weight=1.0)
                        print(node, "<->", de, 1.0, file=nodes_file)

print(graph.number_of_edges())

between = open("offensive_dataset.bet1_2", "w")

for en in english_graph.nodes:
    for de in german_graph.nodes:
        if graph.has_edge(en, de):
            print(en_nodes[en], de_nodes[de], file = between)

en_features = open("offensive_dataset.feat1", "w")
de_features = open("offensive_dataset.feat2", "w")

for node in english_graph.nodes:
    embedding = get_word_embedding(node, english_model, 'english')
    line = str(en_nodes[node])
    if embedding is None:
        embedding = [0 for i in range(100)]
    for i in embedding:
        line += " "
        line += str(i)
    line += " "
    if node in en_dict or len(node.split()) > 1:
        line += "1"
    else:
        line += "0"
    print(line, file = en_features)

for node in german_graph.nodes:
    embedding = get_word_embedding(node, german_model, 'german')
    line = str(de_nodes[node])
    if embedding is None:
        embedding = [0 for i in range(100)]
    for i in embedding:
        line += " "
        line += str(i)
    line += " "
    offensive = False
    for key, value in en_dict.items():
        for v in value:
            if node == v.replace("-", " ").replace("_", " "):
                offensive = True
    if len(node.split()) > 1:
        offensive = True
    
    if offensive:
        line += "1"
    else:
        line += "0"
    
    print(line, file = de_features)

