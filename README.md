# Caregiver Chatbot: Topic Modeling with spaCy

## Overview

This script is an integral part of a graduation project prototype, which aims to develop a chatbot functioning as a caregiver for individuals with dementia using OpenAI LLMS and NLP. Specifically, this code is designed to analyze the conversations between patients and the chatbot, extracting topics and determining the sentiment associated with each topic. The results are stored in a dataset that serves as a feedback loop, guiding the chatbot on which topics to avoid in future conversations.

## Dependencies

1. **spaCy**: Used for various NLP tasks including tokenization and lemmatization.
2. **TextBlob**: Aids in determining the sentiment of the text.
3. **Autocorrect**: Used for spelling correction.
4. **nltk**: Required for tokenization.
5. **neuralcoref**: An add-on for spaCy, used for coreference resolution.

## Key Functionalities

- **Coreference Resolution**: Pronouns in the conversation are replaced with the nouns they refer to, making the text more explicit.
- **Topic Extraction**: Topics are extracted by identifying nouns in the conversation.
- **Sentiment Analysis**: Sentiment associated with each topic is determined using TextBlob. This sentiment can be positive, negative, or neutral.
- **Time Decay Mechanism**: The sentiment associated with each topic is adjusted over time using a decay factor. This means that as time passes, older sentiments have less influence on the overall sentiment of the topic.

## Code Structure

- **Initialization**: The script starts by loading the required spaCy model and adding `neuralcoref` to its pipeline.
- **Patient Class**: Represents a patient and stores their sentiment history related to various topics.
- **Helper Functions**: These include functions for coreference resolution, spelling correction, lemmatization, topic extraction, sentiment determination, and conversation processing.
- **Main Logic**: Processes the patient's response, extracts topics, determines sentiment, and updates the sentiment history for each topic.
- **Decay Mechanism**: Older sentiments are weighted less than recent sentiments using a decay factor. This ensures the chatbot's feedback remains relevant and up-to-date.

![topic modelling](https://github.com/laithab90/topic-modeling_spacy/assets/95342563/1888112e-7f1b-4703-9273-a84684c3ed06)

![timedecay](https://github.com/laithab90/topic-modeling_spacy/assets/95342563/c7391396-d512-4f1d-8438-d9922b0f08b7)

## Usage

1. Ensure all dependencies are installed.
2. Provide the conversation text and path to save sentiment history.
3. Run the script. It will process the conversation, extract topics, determine sentiment, and save the sentiment history.
4. The function `get_topics_to_avoid` can be called to determine which topics the chatbot should avoid in future based on past sentiments.

## Conclusion

This script plays a pivotal role in the caregiver chatbot project by offering insights into how patients feel about certain topics. By avoiding negative topics, the chatbot can provide a more pleasant and supportive interaction for dementia patients.



