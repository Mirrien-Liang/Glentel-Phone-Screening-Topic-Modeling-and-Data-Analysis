"""
This module contains functions for preprocessing text data.
"""

import pandas as pd
import string
import re
import nltk
from langdetect import detect
from translate import Translator
from tqdm import tqdm
from dotenv import load_dotenv
from utils.translation import translate
from utils.spellcheck import spellcheck
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from config import CLEANED_DATA_FILE_PATH, COLUMNS_OF_INTEREST, HAS_PROXY, RESPONSE_COLUMN, CUSTOM_STOPWORDS, CUSTOM_SYNONYMS_DICT, RESPONSE_COLUMN_LEVEL, TO_TRANSLATE
import os

def clean_text(text):
    """
    Function to clean the text data.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    
    Example:
        >>> text = "This is a test text. It contains some special characters like !@#$%^&*() and numbers like 12345."
        >>> clean_text(text)
        'this is a test text contains special characters numbers'
    
    Note:
        This function performs basic cleaning operations on the text data. It can be modified to perform more advanced cleaning operations.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", "", text)
    # Remove numbers - if they are not important for your analysis
    text = re.sub(r"\d+", "", text)
    # Remove whitespace
    text = " ".join(text.split())  # Remove extra whitespace
    return text

# Function to convert nltk POS tags to wordnet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def tokenize_and_lemmatize(text, lemmatizer, stopwords, synonyms=CUSTOM_SYNONYMS_DICT):
    """
    Function to tokenize and lemmatize the text.
    
    Args:
        text (str): The text to tokenize and lemmatize.
        lemmatizer (WordNetLemmatizer): The WordNetLemmatizer object to use for lemmatization.
        stopwords (set): The set of stopwords to remove from the text. Default is the set of English stopwords.
        synonyms (dict): The dictionary of synonyms to merge words with similar meanings.
    
    Returns:
        list: A list containing the tokenized and lemmatized text with stopwords removed.
    
    Example:
        >>> text = "graduated carefully students have carefully wisely graduated are graduating from graduated schools"
        >>> lemmatizer = WordNetLemmatizer()
        >>> tokenize_and_lemmatize(text, lemmatizer, english_stopwords, CUSTOM_SYNONYMS_DICT)
        ['student', 'carefully', 'school', 'wisely', 'graduate']

    Note:
        The lemmatizer is used to reduce words to their base or root form. For example, "running" and "runs"
        will be reduced to "run".
    """
    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords
    tagged = nltk.pos_tag(tokens)

    # Lemmatize text
    lemmatized_text = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged]

    # Remove stopwords and map synonyms
    lemmatized_text = [synonyms.get(word, word) for word in lemmatized_text if not word in stopwords and len(word) > 2] # Remove all 2-char words

    return list(set(lemmatized_text)) # Remove duplicates in themes


def preprocess_data(data):
    """
    Function to preprocess the phone screen data and save it to a CSV file.

    Cleaning Order:
        1. (Disabled) Detect language
        2. (Disabled) Spellcheck
        3. Translate all from French to English
        4. Remove punctuation, stopwords, numbers, whitespace, and lowercase the text
        5. Tokenize and lemmatize the text
        6. Save the processed data to a CSV file
        7. Return the DataFrame
    
    Args:
        data (pd.DataFrame): The input data.

    Returns:
        data (pd.DataFrame): The preprocessed data.
    """

    # Ensure that NLTK resources are downloaded (if not already available)
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download('averaged_perceptron_tagger')

    # Initialize the Lemmatizer and Translator objects
    lemmatizer = WordNetLemmatizer()

    # Get stopwords once
    english_stopwords = set(stopwords.words('english'))

    if TO_TRANSLATE:
        if not HAS_PROXY:
            translator = Translator(from_lang="fr", to_lang='en')

        else:
            # Initialize proxies for translator
            proxies = []
            load_dotenv()
            PROXY_USERNAME = os.getenv("PROXY_USERNAME")
            PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")
            PROXY_HOST = os.getenv("PROXY_HOST")
            PROXY_PORT = os.getenv("PROXY_PORT")

            ###################################################
            #######  Update IP Consistency Policy Below #######
            ###################################################
            for i in range(1, 10000):
                proxies.append(f"://{PROXY_USERNAME}-session-{i}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}")
            ###################################################
            #######  Update IP Consistency Policy Above #######
            ###################################################

            translator = Translator(
                from_lang="fr", # Hard-coded to French
                to_lang='en',
                proxies=proxies
            )
    
    # Initialize English stopwords set
    english_stopwords = set(stopwords.words('english'))
    modified_stopwords = set()

    # Remove punctuation from stopwords (e.g., "aren't" becomes "arent")
    for word in english_stopwords:
        modified_stopwords.add(word.replace("'", ""))
    modified_stopwords.update(CUSTOM_STOPWORDS)

    # Perform basic cleaning operations on each column of interest
    for column in tqdm(COLUMNS_OF_INTEREST, desc='Column Loop', position=0):
        # Report non-existing columns
        if column not in data.columns:
            # raise ValueError(f"Column {column} not found in data.")
            print(f"Warning: {column} not found in data. Skipping.")
            continue
        
        # Create new columns for theme and sentiment
        data[f"cleaned_{column}"] = pd.Series(dtype='object')

        # Loop through each row of the data
        for i, row in tqdm(data.iterrows(), total=data.shape[0], desc='Row Loop', leave=False, position=1):
            # Get the text from the current row
            text = row[column]

            # If text is empty, just copy it over to the new columns
            if pd.isna(text) or text.strip() == "":
                # If text is NaN or empty, just copy it over to the new columns
                data.at[i, f"cleaned_{column}"] = text
                continue
            
            # Step 1: (Disabled) Detect language
            # if TO_TRANSLATE and HAS_PROXY:
            #     lang = detect(text)
            #     if lang != "en":
            #         text = translate(translator, text)

            # Step 2: Translate in all cases
            # Spelling error ignored.
            # Always assuming French.
            # If given in English, no effect.
            if TO_TRANSLATE:
                if HAS_PROXY:
                    text = translate(translator, text)
                else:
                    if detect(text) != "en":
                        text = translate(translator, text)

            # Step 2.5: Spellcheck (Performance to be evaluated)
            text = spellcheck(text)

            # Step 3: Clean the text
            text = clean_text(text)
            
            # Step 4: Tokenize and lemmatize for sentiment analysis (with and without stopwords)
            theme_tokens = tokenize_and_lemmatize(text, lemmatizer, modified_stopwords, CUSTOM_SYNONYMS_DICT)
            
            # Step 5: Join spellchecked tokens and assign to new columns
            data.at[i, f"cleaned_{column}"] = ' '.join(theme_tokens)

    # Step 6: Categorize the key response column and merge "Maybe" and "No" into Not-"Yes" category
    data[f"IND_{RESPONSE_COLUMN}"] = data[RESPONSE_COLUMN].apply(lambda x: 1 if RESPONSE_COLUMN_LEVEL[0] in x else 0)

    # Step 7: Save the preprocessed data to a CSV file (to reduce resource waste)
    data.to_csv(CLEANED_DATA_FILE_PATH, index=False, encoding="utf-8")
    return data

# Example usage:
if __name__ == "__main__":
    preprocess_data()
