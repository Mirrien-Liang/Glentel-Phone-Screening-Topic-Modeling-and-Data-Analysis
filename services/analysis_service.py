import numpy as np
import matplotlib.pyplot as plt


def interpret_coherence(coherence_score):
    if coherence_score > 0.6:
        return "high coherence indicating clear and interpretable topics."
    elif coherence_score > 0.4:
        return "moderate coherence with room for improvement in topic clarity."
    else:
        return "low coherence, suggesting topics may be less meaningful or distinct."


def parse_confusion_matrix(cm_lines):
    # Remove the brackets and split the remaining string into numbers
    cm_lines = [line.strip("[] ") for line in cm_lines]
    cm_list = [list(map(int, line.split())) for line in cm_lines if line]

    # Check if the confusion matrix has the correct format (2x2 for binary classification)
    if len(cm_list) != 2 or any(len(row) != 2 for row in cm_list):
        raise ValueError("The confusion matrix is not in the expected format of 2x2.")

    cm = np.array(cm_list)
    return cm


def summarize_rf_metrics(metrics_text):
    # Extracting relevant metrics from the classification report
    report_lines = metrics_text.split("\n")
    accuracy_line = [line for line in report_lines if "accuracy" in line][0]
    accuracy = accuracy_line.split()[1]

    # Extracting the confusion matrix from the metrics text
    cm_start = report_lines.index("Confusion Matrix:") + 1  # Start of the matrix
    cm_end = cm_start + 2  # End of the matrix
    cm = parse_confusion_matrix(report_lines[cm_start:cm_end])

    true_negatives, false_positives, false_negatives, true_positives = cm.ravel()

    summary = (
        f"The model correctly predicted {true_positives} positive and "
        f"{true_negatives} negative results. "
        f"There were {false_positives} false positives and "
        f"{false_negatives} false negatives. "
        f"The overall accuracy of the model is {int(float(accuracy)*100)}%.\n\n"
        "* For a detailed report of the random forest model, please refer to the 'metrics.txt' file in the 'models/random_forest/' directory."
    )
    return summary


def summarize_logistic_regression_metrics(metrics_text):
    # Extracting relevant metrics from the classification report
    report_lines = metrics_text.split("\n")
    accuracy_line = [line for line in report_lines if "accuracy" in line][0]
    accuracy = accuracy_line.split()[1]

    # Extracting the confusion matrix from the metrics text
    cm_start = report_lines.index("Confusion Matrix:") + 1  # Start of the matrix
    cm_end = cm_start + 2  # End of the matrix
    cm = parse_confusion_matrix(report_lines[cm_start:cm_end])

    true_negatives, false_positives, false_negatives, true_positives = cm.ravel()

    summary = (
        f"The model correctly predicted {true_positives} positive and "
        f"{true_negatives} negative results. "
        f"There were {false_positives} false positives and "
        f"{false_negatives} false negatives. "
        f"The overall accuracy of the model is {int(float(accuracy)*100)}%.\n\n"
        "* For a detailed report of the logistic regression model, please refer to the 'metrics.txt' file in the 'models/logistic_model/' directory."
    )

    return summary


def plot_combined_lift_chart(cumulative_gains_rf, cumulative_gains_lr):
    plt.figure(figsize=(12, 10))
    plt.plot(cumulative_gains_rf, label='Random Forest')
    plt.plot(cumulative_gains_lr, label='Logistic Regression')
    plt.plot(np.linspace(0, 1, len(cumulative_gains_rf)), label='Baseline', linestyle='--')
    plt.title('Combined Lift Chart')
    plt.xlabel('Proportion of sample')
    plt.ylabel('Cumulative gain')
    plt.legend()
    plt.tight_layout()
    
    combined_lift_chart_path = 'results/combined_lift_chart.png'
    plt.savefig(combined_lift_chart_path, bbox_inches='tight')
    plt.close()
    
    return combined_lift_chart_path


def generate_report(
    coherence_filepath="models/lda_model/coherence_scores.txt",
    rf_metrics_filepath="models/random_forest/metrics.txt",
    lr_metrics_filepath="models/logistic_model/metrics.txt",
):
    # Read coherence scores and interpret them
    with open(coherence_filepath, "r") as f:
        coherence_lines = f.readlines()
    coherence_interpretations = [
        f"{line.strip()} - {interpret_coherence(float(line.split(': ')[1]))}"
        for line in coherence_lines
    ]
    # Read random forest metrics and summarize them
    with open(rf_metrics_filepath, "r") as f:
        rf_metrics_text = f.read()
    rf_summary = summarize_rf_metrics(rf_metrics_text)

    # Read logistic regression metrics and summarize them
    with open(lr_metrics_filepath, "r") as f:
        lr_metrics_text = f.read()
    lr_summary = summarize_logistic_regression_metrics(lr_metrics_text)

    # Generate the report content
    report_content = (
        "Topic Modeling Coherence Scores:\n"
        + "\n".join(coherence_interpretations)
        + "\n\nRandom Forest Model Summary:\n"
        + rf_summary
        + "\n\nLogistic Regression Model Summary:\n"
        + lr_summary
    )

    with open("results/Model_Report.txt", "w") as f:
        f.write(report_content)

    return report_content


if __name__ == "__main__":
    generate_report()
