# topic_extract.py

import pandas as pd
import pickle
from tqdm import tqdm
from gensim import corpora
from gensim.models import LdaModel
from models.topic_evaluate import evaluate_model
from config import MIN_FREQ_TO_KEEP_WORDS

class TopicExtraction:
    def __init__(self, data, question, passes, num_topics_start, num_topic_stop, num_trains_per_step=3, step_size=1):

        self.data = data
        self.question = question
        self.num_topics_start = num_topics_start
        self.num_topics_stop = num_topic_stop
        self.num_topics_stop_adjusted = num_topic_stop + 1 if (num_topic_stop - num_topics_start) % step_size == 0 else num_topic_stop + (step_size - (num_topic_stop - num_topics_start) % step_size)
        self.passes = passes
        self.num_trains_per_step = num_trains_per_step
        self.step_size = step_size
        self.lda_model = None
        self.dictionary = None
        self.documents = None
        self.corpus = None
        self.num_topics = None
        

    def process_question(self):
        # Prepare documents and dictionary for the question
        documents = [doc.split() for doc in self.data[f"cleaned_{self.question}"].dropna() if doc.strip()]
        dictionary = corpora.Dictionary(documents)
        dictionary.filter_extremes(no_below=MIN_FREQ_TO_KEEP_WORDS, no_above=0.5, keep_n=1000)
        
        # Prepare the corpus for the question
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        
        # Train X LDA models for the question and pick the best one
        max_coherence_score = 0
        for num_topics in tqdm(range(self.num_topics_start, self.num_topics_stop_adjusted, self.step_size), desc= "Num_Topic Stepping", leave=False, position=1):
            for _ in tqdm(range(self.num_trains_per_step), desc= "In-step Iterating", leave=False, position=2):
                try:
                    lda_model = LdaModel(corpus=corpus,
                                        id2word=dictionary,
                                        num_topics=num_topics,
                                        passes=self.passes,
                                        # random_state=42,
                                        )
                    # Evaluate the model
                    coherence_score = evaluate_model(lda_model, dictionary, documents, self.question)
                    if coherence_score > max_coherence_score:
                        max_coherence_score = coherence_score
                        best_lda_model = lda_model
                        best_dictionary = dictionary
                        best_corpus = corpus
                        best_documents = documents
                        best_num_topics = num_topics
                except ValueError:
                    pass


        # print(f'"{self.question}" Coherence Score:', max_coherence_score)

        # Save the model, dictionary, corpus, and documents
        best_lda_model.save(f'models/lda_model/{self.question.replace("?","")}_lda.model')
        best_dictionary.save(f'models/lda_model/{self.question.replace("?","")}_dictionary.dict')
        with open(f'models/lda_model/{self.question.replace("?","")}_corpus.pkl', 'wb') as f:
            pickle.dump(best_corpus, f)
        with open(f'models/lda_model/{self.question.replace("?","")}_texts.pkl', 'wb') as f:
            pickle.dump(best_documents, f)

        # Create and save the topic dictionary mapping
        self.save_topic_dictionary(
            best_lda_model,
            best_dictionary,
            f'models/lda_model/{self.question.replace("?","")}_topic_dictionary.txt',
            save_prob= True
        )

        self.save_topic_dictionary(
            best_lda_model,
            best_dictionary,
            f'results/{self.question.replace("?","")}_topic_dictionary.txt',
            save_prob = False
        )

        # Self Assignments
        self.lda_model = best_lda_model
        self.dictionary = best_dictionary
        self.corpus = best_corpus
        self.documents = best_documents
        self.num_topics = best_num_topics


    def save_topic_dictionary(self, lda_model, dictionary, file_path, save_prob):
        topic_descriptions = {}
        for topic_id in range(lda_model.num_topics):
            # Extract the top words for the topic
            top_words = lda_model.show_topic(topic_id, topn=10)
            # Combine the words with their probabilities
            # Using direct dictionary access instead of id2token
            topic_description = {word_id: prob for word_id, prob in top_words if word_id in dictionary.token2id}

            # Add to the overall topic_descriptions dictionary
            topic_descriptions[topic_id] = topic_description
        
        # Save the topic descriptions dictionary to a file
        with open(file_path, 'w', encoding='utf-8') as file:
            if save_prob:
                for topic_id, words in topic_descriptions.items():
                    words_str = ', '.join([f"{word} ({prob:.3f})" for word, prob in words.items()])
                    file.write(f"Topic {topic_id + 1}: {words_str}\n")
            else:
                for topic_id, words in topic_descriptions.items():
                    words_str = ', '.join([word for word in words])
                    file.write(f"Topic {topic_id + 1}: {words_str}\n")


if __name__ == '__main__':
    COLUMNS_OF_INTEREST = [
        "Q1",
        "Q2",
        "Q3",
    ]
    data = pd.read_csv('data/processed/cleaned_data.csv')
    topic_extractor = TopicExtraction(data=data, question=COLUMNS_OF_INTEREST[0], passes=25, num_topics_start=2, num_topic_stop=15, num_trains_per_step=3, step_size=1)
    topic_extractor.process_question()
