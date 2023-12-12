import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


# Function to prepare the training and testing data with one-hot encoding
def prepare_data(data, features, target):
    # Replace 'NA' with -1 for topic assignment features
    for feature in features:
        data[feature] = data[feature].replace('NA', -1).fillna(-1).astype(int)
    
    # One-hot encode the features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(data[features])
    
    # Encode the target variable (Merging 'maybe' and 'no')
    # data[target] = data[target].replace({'-1': 0}).fillna(0).astype(int)
    
    X = encoded_features
    y = data[target]
    feature_names = encoder.get_feature_names_out(features)
    return train_test_split(X, y, test_size=0.5, random_state=42), feature_names, encoder

# Function to train the Random Forest model
def train_random_forest(X_train, y_train, n_estimators=500, max_features='sqrt'):
    model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model to a file
    try:
        with open("models/random_forest/model.pkl", 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        print(f"Error saving model: {e}")

    return model

# Function to evaluate the model and write the results to a file
def evaluate_model(model, X_test, y_test, feature_names, output_dir):
    predictions = model.predict(X_test)
    prediction_probs = model.predict_proba(X_test)

    # Write classification report and confusion matrix to a file
    cm = confusion_matrix(y_test, predictions)
    cr = classification_report(y_test, predictions, output_dict=True)

    # Write classification report and confusion matrix to a file with interpretations
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, predictions))

        # Interpretation for classification report
        f.write("\nThe model's accuracy is {:.2f}% which reflects the overall rate of correct predictions. "
                "Precision tells us the proportion of positive identifications that were actually correct, "
                "and recall tells us the proportion of actual positives that were identified correctly.\n".format(cr['accuracy']*100))

        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nThe confusion matrix shows the counts of correct and incorrect predictions made by the model. "
                "It's a table with two rows and two columns that reports the number of false positives, "
                "false negatives, true positives, and true negatives. This helps to understand the model's performance in terms of specificity (true negative rate) and sensitivity (true positive rate).\n")
        
    # Additional insights
    model_confidence = np.max(prediction_probs, axis=1).mean()  # Average confidence of the model in its predictions
    error_indices = np.where(predictions != y_test)[0]  # Indices where the model predictions were incorrect
    class_balance = np.mean(y_test)  # Assuming binary classification for simplicity

    # Write insights to a file
    with open(metrics_path, 'a') as f:
        f.write(f"\nModel Confidence: On average, the model is {model_confidence:.2%} confident in its predictions.\n")
        f.write(f"\nError Analysis: The model made incorrect predictions for {len(error_indices)} out of {len(y_test)} samples.\n")
        f.write(f"\nClass Balance Impact: The dataset has {class_balance:.2%} positive instances.\n")


    # Feature Importance Plot for one-hot encoded features
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(16, 10))  # Adjust the figure size as needed
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=8)
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
    plt.title('Feature Level Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature Levels')

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight')
    plt.savefig('results/random_forest_feature_importance.png', bbox_inches='tight')
    plt.close()


    # Save feature importance scores and feature level names to metrics.txt
    with open(metrics_path, 'a') as f:
        f.write("\n\nFeature Level Importance:\n")
        for i in indices:
            f.write(f"{feature_names[i]}: {importance[i]:.4f}\n")

    # Lift Chart (Cumulative Gains Chart)
    proba = model.predict_proba(X_test)[:, 1]
    sorted_index = np.argsort(proba)[::-1]
    y_test_sorted = np.array(y_test)[sorted_index]  # Convert to numpy array before indexing
    cumulative_gains = np.cumsum(y_test_sorted) / np.sum(y_test_sorted)
    plt.figure(figsize=(12, 10))
    plt.plot(cumulative_gains, label='Random Forest')
    plt.plot(np.linspace(0, 1, len(cumulative_gains)), label='Baseline', linestyle='--')
    plt.title('Random Forest Lift Chart')
    plt.xlabel('Proportion of sample')
    plt.ylabel('Cumulative gain')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lift_chart.png'), bbox_inches='tight')
    # plt.savefig('results/random_forest_lift_chart.png', bbox_inches='tight')
    plt.close()
    return cumulative_gains

# Function to make predictions
def make_predictions(model, encoder, data, features, output_csv = 'results/dataset_with_predictions.csv'):
    # Replace 'NA' with -1 for topic assignment features
    for feature in features:
        data[feature] = data[feature].replace('NA', -1).fillna(-1).astype(int)
    
    # Apply one-hot encoding using the trained encoder
    encoded_features = encoder.transform(data[features])
    
    # Make predictions using the Random Forest model
    predictions = model.predict(encoded_features)
    
    # Add predictions to the dataset
    data['Predicted'] = predictions
    
    # Save the dataset with predictions to a CSV file
    # data.to_csv(output_csv, index=False)

    return data


# Main function to run the model training and evaluation
def main(data, features, target, output_dir = "models/random_forest"):
    (X_train, X_test, y_train, y_test), feature_names, encoder = prepare_data(data, features, target)
    model = train_random_forest(X_train, y_train)
    rf_cumulative_gain = evaluate_model(model, X_test, y_test, feature_names, output_dir)
    data = make_predictions(model, encoder, data, features)
    return data, rf_cumulative_gain


if __name__ == '__main__':
    # Paths and other constants
    data = pd.read_csv('data/processed/cleaned_data_with_dominant_topics.csv')
    # Feature list
    features = [
        "Q1",
        "Q2",
        "Q3",
    ]
    target = "TARGET"
    output_dir = 'models/random_forest'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run the main function
    main(data, features, target, output_dir)