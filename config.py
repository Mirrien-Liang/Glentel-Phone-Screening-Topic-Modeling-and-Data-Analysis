"""
Define all the global configurations here, like file paths, language options, constants for data processing, etc.
"""


"""
You have to configure the following items properly before execution.
"""

# Path to the raw data file (Excel or CSV)
RAW_DATA_FILE_PATH = "data/raw/sample_raw_data.csv"

# To preprocess the dataset?
TO_PREPROCESS_DATA = True

# If to preprocess, set True to translate
TO_TRANSLATE = True

# If to translate, set True to use proxy (HIGHLY RECOMMENDED)
# But if no proxy, set false to auto-detect language (with strict usage limit imposed)
# In this script we used BrightData Proxy
HAS_PROXY = False 

# To run topic extraction pipeline?
# If not, run predict models directly.
TO_EXTRACT_TOPICS = True

# Columns of interest
COLUMNS_OF_INTEREST = ["Q1","Q2","Q3"]

# Column to predict and compare
RESPONSE_COLUMN = "TARGET"
RESPONSE_COLUMN_LEVEL = ['Yes', 'No'] # Must be exactly the same; assumed binary; place the level to predicted first;


# Adjust specific keywords you wish to remove in addition to traditional English stopwords
CUSTOM_STOPWORDS = ["etc", "eg", "ie", "im", "", "around", "about", "already", "yet", "’", "therefore", "although", "though", "despite", "while", "whereas", "however", "but", "really", "certain", "much", "many", "most", "actually", "never", "pretty", "quite", "may", "might", "thing", "something", "anything", "someone", "anyone", "inside", "outside", "hopefully", "ideally", "likely", "probably", "possibly", "usually", "casually", "even", "really", "enough", "specifically", "just", "previously", "basically", "definitely", "another", "other", "others", "would", "every", "think", "always", "rarely", "barely", "usually", "instead", "across", "honestly", "especially", "personally", "mainly", "guy", "guys", "recently", "lately", "yes", "currently" , "mainly", "originally", "surely", "mostly", "since", "sure", "seem", "bit", "also", "several", "regard", "somewhere", "eventually", "reason", "see",]


# Synonyms mapping dictionary
CUSTOM_SYNONYMS_DICT = {
    "manager": "management",
    "manage": "management",
    "supervisor": "management",
    "leader": "leadership",
    "lead": "leadership",
    "organization": "company",
    "relocate": "move",
    "relocation": "move",
    "role": "position",
    "cell": "phone",
    "mobile": "phone",
    "sell": "sale",
    "salary": "pay",
    "wage": "pay",
    "well": "good",
    # Add more mappings as needed
}


"""
You might want to modify below with more caution.
"""


# Desired numbers of topics in each set of topics in Topic Extraction
# Numbers in order of the COLUMNS_OF_INTEREST
NUM_OF_TOPIC_START = [2, 2, 2]
NUM_OF_TOPIC_STOP = [10, 10, 10]
STEP_SIZE = [1, 1, 1]
NUM_OF_TRAIN_PER_STEP = [3, 3, 3]
NUM_OF_PASSES = [25, 25, 25] # Number of passes through the corpus in the LDA model
MIN_FREQ_TO_KEEP_WORDS = 1 # Minimum frequency of words to keep in the dictionary (Increase as dataset gets larger, here for such a small sample we used 1)

# If to preprocess the dataset, provide the path to store the cleaned data CSV file
CLEANED_DATA_FILE_PATH = "data/processed/cleaned_data.csv"

# Path to store the topic assigned data CSV file
TOPIC_ASSIGNMENT_FILE_PATH = "data/processed/cleaned_data_with_dominant_topics.csv"

# Path to store the processed data CSV file
PROCESSED_DATA_FILE_PATH = "data/processed/processed_data.csv"