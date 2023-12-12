import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample


def prepare_data(data, features, target):
    # Replace 'NA' with -1 for topic assignment features
    for feature in features:
        data[feature] = data[feature].replace("NA", -1).fillna(-1).astype(int)

    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoded_data = encoder.fit_transform(data[features]).toarray()
    encoded_feature_names = encoder.get_feature_names_out(features)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names)

    # Combine the encoded features with the target
    data_prepared = pd.concat([encoded_df, data[target]], axis=1)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_prepared.iloc[:, :-1],
        data_prepared[target],
        test_size=0.5,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test, data_prepared, encoded_feature_names


def model_precheck(data_prepared, features, target):
    # Check for independence between features (correlation plot)
    plt.figure(figsize=(20, 16))
    sns.heatmap(data_prepared.iloc[:, :-1].corr(), annot=True, fmt=".2f")
    plt.title("Correlation Plot of Features")
    plt.savefig("models/logistic_model/correlation_plot.png")
    plt.close()

    # Check for sample size adequacy
    num_events = data_prepared[target].sum()
    num_features = len(features)
    if num_events < 10 * num_features:
        print(
            f"Warning: Not enough events for the number of features. Events: {num_events}, Features: {num_features}"
        )

    # Write the pre-check results to a file
    with open("models/logistic_model/pre_checks_report.txt", "w") as f:
        f.write(f"Check Correlation Plot in the same folder.\n")
        f.write(
            f"Sample Size Check {'PASS' if num_events >= 10 * num_features else 'FAILED'}: Events - {num_events}, Features - {num_features}\n"
        )


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    with open("models/logistic_model/metrics.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nAccuracy: {accuracy*100:.2f}%\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(matrix))
        f.write("\n\n")

    # Get the probability predictions for the positive class (assuming binary classification)
    proba = model.predict_proba(X_test)[:, 1]

    # Sort the indices in descending order of the probabilities
    sorted_index = np.argsort(proba)[::-1]

    # Sort the true labels according to the sorted indices
    y_test_sorted = np.array(y_test)[sorted_index]

    # Calculate the cumulative gain
    cumulative_gains = np.cumsum(y_test_sorted) / np.sum(y_test_sorted)

    # Create the plot
    plt.figure(figsize=(12, 10))
    plt.plot(cumulative_gains, label="Logistic Regression")
    plt.plot(np.linspace(0, 1, len(cumulative_gains)), label="Baseline", linestyle="--")
    plt.title("Logistic Regression Lift Chart")
    plt.xlabel("Proportion of sample")
    plt.ylabel("Cumulative gain")
    plt.legend()
    plt.tight_layout()

    # Save the lift chart
    plt.savefig(
        "models/logistic_model/logistic_regression_lift_chart.png", bbox_inches="tight"
    )
    # plt.savefig('results/logistic_regression_lift_chart.png', bbox_inches='tight')

    # Close the plot
    plt.close()
    return cumulative_gains


def bootstrap_feature_significance(
    X, y, n_iterations=100, model=LogisticRegression(max_iter=1000)
):
    """
    Estimates the significance of features using bootstrapping.
    """
    coef_distributions = []

    for i in range(n_iterations):
        # Bootstrap sample
        X_sample, y_sample = resample(X, y)

        # Fit model
        model.fit(X_sample, y_sample)

        # Store coefficients
        coef_distributions.append(model.coef_[0])

    # Convert to DataFrame for analysis
    coef_df = pd.DataFrame(coef_distributions, columns=X.columns)

    return coef_df


def interpret_model(model, feature_names, X_train, y_train):
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": coefficients}
    )
    feature_importance = feature_importance.sort_values(
        by="Coefficient", ascending=False
    )

    with open("models/logistic_model/metrics.txt", "a") as f:
        f.write("Feature Importance:\n")
        f.write(feature_importance.to_string(index=False))
        f.write("\n\n")

    # Create a horizontal bar plot
    plt.figure(figsize=(24, 12))  # Adjust the size as needed
    plt.barh(
        feature_importance["Feature"], feature_importance["Coefficient"], color="b"
    )
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Features")
    plt.title("Feature Importance in Logistic Regression Model")
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
    plt.tight_layout()

    # Save the plot
    plt.savefig("models/logistic_model/feature_importance.png")
    # plt.savefig("results/logistic_feature_importance.png")
    plt.close()

    # Bootstrapping for feature significance
    coef_df = bootstrap_feature_significance(X_train, y_train)
    coef_df_mean = coef_df.mean().sort_values(ascending=True)
    coef_df_std = coef_df.std()

    # Plotting the bootstrap coefficients
    plt.figure(figsize=(16, 12))  # Adjust figure size as necessary
    # Create horizontal bar plot
    plt.barh(
        coef_df_mean.index, coef_df_mean, xerr=coef_df_std
    )  # Changed to barh for horizontal plot
    plt.title("Feature Coefficients with Bootstrapping")
    plt.ylabel("Features")  # Swapped x and y labels
    plt.xlabel("Coefficient Value")
    plt.tight_layout()  # Ensure everything fits without overlapping
    plt.savefig("models/logistic_model/feature_significance.png")
    plt.savefig("results/feature_significance.png")

    return feature_importance


def make_predictions(model, X):
    return model.predict(X)


def main(data, features, target):
    # Preparing data and performing pre-checks
    (
        X_train,
        X_test,
        y_train,
        y_test,
        data_prepared,
        encoded_feature_names,
    ) = prepare_data(data, features, target)
    model_precheck(data_prepared, features, target)

    # Train and evaluate the model
    model = train_logistic_regression(X_train, y_train)
    lr_cumulative_gain = evaluate_model(model, X_test, y_test)

    # Interpret the model
    interpret_model(model, encoded_feature_names, X_train, y_train)
    # print(feature_importance)

    # Optionally, use the model to make predictions
    return data_prepared, lr_cumulative_gain


if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv("data/processed/cleaned_data_with_dominant_topics.csv")

    # Feature list
    features = [
        "Q1",
        "Q2",
        "Q3",
    ]
    target = "TARGET"

    main(data, features, target)
