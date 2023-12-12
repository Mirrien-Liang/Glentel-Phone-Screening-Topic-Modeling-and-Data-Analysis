# topic_assign.py

import pandas as pd
from config import *
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary


def assign_topics(lda_model, dictionary, document):
    if pd.isna(document) or document.strip() == '':
        return 'NA'
    tokens = document.split()
    bow = dictionary.doc2bow(tokens)
    topics = lda_model.get_document_topics(bow)
    dominant_topic = sorted(topics, key=lambda x: x[1], reverse=True)[0][0] + 1
    return dominant_topic


def assign_topics_to_data(data, lda_model, dictionary, question):
    data[f"{question}_dominant_topic"] = data[f"cleaned_{question}"].apply(lambda doc: assign_topics(lda_model, dictionary, doc))
    return data


def prepend_id_counts(data, question):
    # Compute the frequency count for all IDs in the column
    freq_counts = data[f"{question}_dominant_topic"].value_counts().to_dict()
    dict_filename = f"models/lda_model/{question.replace('?','')}_topic_dictionary.txt"

    # Read the dictionary file and prepend counts
    with open(dict_filename, 'r') as file:
        dictionary_lines = file.readlines()

    # Update dictionary lines with counts
    updated_lines = []
    for idx, meaning in enumerate(dictionary_lines, start=1):
        count = freq_counts.get(idx, 0)  # Default to 0 if ID is not found
        updated_lines.append(f"[{count} responses] {meaning.strip()}")

    # Write the results back to the file or return the updated lines
    with open(dict_filename, 'w') as file:
        file.writelines('\n'.join(updated_lines))

    with open(f"results/{question.replace('?','')}_topic_dictionary.txt", 'w') as file:
        file.writelines('\n'.join(updated_lines))

    return updated_lines


if __name__ == "__main__":
    def load_model_and_dictionary(question_name):
        lda_model = LdaModel.load(f'models/lda_model/{question_name.replace("?","")}_lda.model')
        dictionary = Dictionary.load(f'models/lda_model/{question_name.replace("?","")}_dictionary.dict')
        return lda_model, dictionary


    def main(data, questions):
        for question in questions:
            lda_model, dictionary = load_model_and_dictionary(question)

            # Make sure we're not trying to strip NaN values
            data['cleaned_' + question] = data['cleaned_' + question].fillna('')  # Replace NaN with empty strings
            
            data = assign_topics_to_data(data, lda_model, dictionary, question)

        output_path = TOPIC_ASSIGNMENT_FILE_PATH
        data.to_csv(output_path, index=False)
        return data
    
    
    COLUMNS_OF_INTEREST = [
        "Q1",
        "Q2",
        "Q3",
    ]
    
    data = pd.read_csv('data/processed/cleaned_data.csv')
    main(data,COLUMNS_OF_INTEREST)