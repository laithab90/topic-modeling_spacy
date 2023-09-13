import json
import pandas as pd
import os
from textblob import TextBlob
from autocorrect import Speller
import nltk
import spacy
import neuralcoref
import os
import pandas as pd
import json
# Load the spaCy model
nlp = spacy.load('en_core_web_sm')
# Add NeuralCoref to spaCy's pipe
neuralcoref.add_to_pipe(nlp)
# Initialize the spell checker
spell = Speller(lang='en')
# Download the Punkt tokenizer for sentence splitting
nltk.download('punkt') 
class Patient:
    def __init__(self):
        self.sentiment_history = {}  # topics are keys, sentiment scores are values

def replace_pronouns_with_nouns(text):
    doc = nlp(text)
    return doc._.coref_resolved  # this attribute contains the text with pronouns replaced

def correct_spelling_and_lemmatize(text):
    #corrected_text = spell(text)
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text



def extract_topics(text):
    doc = nlp(text)
    # Just using nouns as topics in this simple example
    return [token.text for token in doc if token.pos_ == 'NOUN']

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def process_conversation(patient, conversation):
    conversation_with_resolved_pronouns = replace_pronouns_with_nouns(conversation)
    corrected_and_lemmatized_conversation = correct_spelling_and_lemmatize(conversation_with_resolved_pronouns)
    return corrected_and_lemmatized_conversation

def handle_patient_response(patient, response):
    # print(f"Patient's response: {response}")
    topics = extract_topics(response)
    # print(f"Identified topics: {topics}")
    sentiment = get_sentiment(response)
    # print(f"Sentiment of response: {sentiment}")

    for topic in topics:
        if topic not in patient.sentiment_history:
            patient.sentiment_history[topic] = []
        patient.sentiment_history[topic].append(sentiment)

    # print(f"Patient's sentiment history: {patient.sentiment_history}")

# Initialize a patient
patient = Patient()

conversation_path = "chatbot/feedbackloop/log_chatgpt_feedbackloop.csv"
sentiment_path = "chatbot/feedbackloop/sentiment_scores.json"
def analyze_conversation_and_save_topics(file_path, json_file_path):
    # If the file doesn't exist, do nothing
    if not os.path.exists(file_path):
        return

    # Load the dataframe from the file
    df = pd.read_csv(file_path, encoding='latin-1')

    # If the JSON file exists, load the existing topics and their scores
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            existing_sentiment_history = json.load(f)
    else:
        existing_sentiment_history = {}

    # Initialize a patient
    patient = Patient()

    # Join all the conversations into a single text, separated by " ||| "
    all_conversations = " ||| ".join(df['patient'])

    # Replace the pronouns in the entire text
    resolved_conversations = process_conversation(patient, all_conversations)

    # Split the text back into individual conversations
    # We assume that the original conversations did not contain the string " ||| "
    resolved_conversations = resolved_conversations.split(" ||| ")

    for conversation in resolved_conversations:
        sentences = nltk.sent_tokenize(conversation)

        for sentence in sentences:
            handle_patient_response(patient, sentence)

    # Get the compound sentiment scores for the topics
    compound_scores = {topic: sum(sentiments)/len(sentiments) for topic, sentiments in patient.sentiment_history.items()}

    # Update the sentiment history for each topic
    for topic, score in compound_scores.items():
        if topic in existing_sentiment_history:
            existing_sentiment_history[topic].append(score)
        else:
            existing_sentiment_history[topic] = [score]

    # Save the sentiment history to the JSON file
    with open(json_file_path, 'w') as f:
        json.dump(existing_sentiment_history, f)
analyze_conversation_and_save_topics(conversation_path, sentiment_path)
def get_topics_to_avoid(json_file_path, decay_factor=0.8):
    # If the JSON file doesn't exist, return an empty list
    if not os.path.exists(json_file_path):
        return []

    # Load the sentiment history from the JSON file
    with open(json_file_path, 'r') as f:
        sentiment_history = json.load(f)

    # Calculate the decayed compound scores for the topics
    decayed_scores = {topic: sum(score * (decay_factor ** i) for i, score in enumerate(scores)) 
                      for topic, scores in sentiment_history.items()}

    # Get the topics with a negative decayed compound score
    negative_topics = {topic: score for topic, score in decayed_scores.items() if score < 0}

    # Sort the negative topics in descending order of their decayed compound scores
    topics_to_avoid = sorted(negative_topics.items(), key=lambda item: item[1])

    # Extract only the topics, not their scores
    topics_to_avoid = [topic for topic, score in topics_to_avoid]

    return topics_to_avoid

topics_to_avoid = get_topics_to_avoid(sentiment_path)
if not topics_to_avoid:
    topics_to_avoid = ['none']

# Convert the list of topics to a comma-separated string
topics_str = ', '.join(topics_to_avoid)

