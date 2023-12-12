from services import data_loader, analysis_service
from config import *
from models.topic_extract import TopicExtraction
from tqdm import tqdm
import models.topic_assign
import models.topic_evaluate
import models.random_forest
import models.logistic_regression


def run_topic_modeling_pipeline(data):
    # Stores coherence scores
    scores = {}

    for i, question in enumerate(tqdm(COLUMNS_OF_INTEREST, desc= "Question Loop", position=0)):
        # Step 1: Topic Extraction
        topic_extractor = TopicExtraction(
            data=data,
            question=question,
            passes=NUM_OF_PASSES[i],
            num_topics_start=NUM_OF_TOPIC_START[i],
            num_topic_stop=NUM_OF_TOPIC_STOP[i],
            num_trains_per_step=NUM_OF_TRAIN_PER_STEP[i],
            step_size=STEP_SIZE[i],
            )
        
        topic_extractor.process_question()

        # Step 2: Topic Assignment and Evaluation
        # Make sure we're not trying to strip NaN values
        data[f"cleaned_{question}"] = data[f"cleaned_{question}"].fillna("")
        data = models.topic_assign.assign_topics_to_data(
            data,
            topic_extractor.lda_model,
            topic_extractor.dictionary,
            question
        )

        models.topic_assign.prepend_id_counts(data, question)

        scores[question] = models.topic_evaluate.evaluate_model(
            topic_extractor.lda_model,
            topic_extractor.dictionary,
            topic_extractor.documents,
            question
        )

    # Step 3: Save the data and scores
    data.to_csv(TOPIC_ASSIGNMENT_FILE_PATH, index=False)
    with open("models/lda_model/coherence_scores.txt", "w") as f:
        for question_name, score in scores.items():
            f.write(f"'{question_name}' Coherence Score: {score}\n")
            print((f"\n'{question_name}' Coherence Score: {score}"))
    print("")
    return data

def run_random_forest(data, features, target):
    return models.random_forest.main(data, features, target)

def run_logistic_regress(data, features, target):
    return models.logistic_regression.main(data, features, target)

if __name__ == "__main__":
    print("================= Start of Program =================")


    print("Loading Data...")
    df = data_loader.load_data(TO_PREPROCESS_DATA, RAW_DATA_FILE_PATH if TO_PREPROCESS_DATA else CLEANED_DATA_FILE_PATH)
    print(f"Data Loaded ({len(df)} x {len(df.columns)})")

    if TO_EXTRACT_TOPICS:
        print("Running Topic Modeling Pipeline...")
        df = run_topic_modeling_pipeline(df)
        print("Topic Extracted")
    else:
        print("Skipping Topic Modeling Data...")
        df = data_loader.load_data(False, TOPIC_ASSIGNMENT_FILE_PATH)
        print("Topic Modeling Data Loaded")

    # Set up feature columns and target column
    features = [f"{col}_dominant_topic" for col in COLUMNS_OF_INTEREST]
    target = f"IND_{RESPONSE_COLUMN}"

    print("Running Random Forest...")
    df, rf_cumulative_gain = run_random_forest(df, features, target)

    print("Running Logistic Regression...")
    _, lr_cumulative_gain = run_logistic_regress(df, features, target)

    # Plot the combined cumulative chart
    analysis_service.plot_combined_lift_chart(rf_cumulative_gain, lr_cumulative_gain)

    # Store output data frame
    df.to_csv(PROCESSED_DATA_FILE_PATH, index=False)

    print("Generating Report...")
    analysis_service.generate_report()

    print("================= End of Program =================")