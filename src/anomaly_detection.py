
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_DIR = "models"
RESULT_DIR = "results/graphs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================
# ISOLATION FOREST TRAINING
# ==========================================

def train_isolation_forest(
    X_train_normal,
    X_test,
    y_test_binary
):

    print("\n" + "=" * 60)
    print("       ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 60)

    # Step 1: Initialize model
    print("\n[1] INITIALIZING ISOLATION FOREST...")

    iso_model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    print("Model initialized successfully.")

    # Step 2: Train using normal traffic only
    print("\n[2] TRAINING ON NORMAL NETWORK TRAFFIC...")

    iso_model.fit(X_train_normal)

    print("Training completed successfully!")

    print(
        "Normal training samples:",
        X_train_normal.shape[0]
    )

    # Step 3: Predict test observations
    print("\n[3] DETECTING ANOMALOUS NETWORK TRAFFIC...")

    # Isolation Forest:
    # +1 = Inlier (Normal)
    # -1 = Outlier (Anomaly)

    raw_predictions = iso_model.predict(X_test)

    y_pred = np.where(
        raw_predictions == -1,
        "Attack",
        "Normal"
    )

    y_true = np.asarray(y_test_binary)

    print("Anomaly predictions generated successfully.")

    # Step 4: Calculate anomaly scores
    print("\n[4] CALCULATING ANOMALY SCORES...")

    anomaly_scores = iso_model.decision_function(X_test)

    # Lower scores indicate more anomalous observations

    # Step 5: Evaluate model
    print("\n[5] MODEL EVALUATION")

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="Attack",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="Attack",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="Attack",
        zero_division=0
    )

    print("\nAccuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1-Score :", round(f1 * 100, 2), "%")

    # Step 6: Classification report
    print("\n[6] CLASSIFICATION REPORT")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=["Normal", "Attack"],
            zero_division=0
        )
    )

    # Step 7: Confusion matrix
    print("\n[7] GENERATING CONFUSION MATRIX...")

    labels = ["Normal", "Attack"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title("Isolation Forest - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "isolation_forest_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.show()

    # Step 8: Anomaly score visualization
    print("\n[8] GENERATING ANOMALY SCORE DISTRIBUTION...")

    plt.figure(figsize=(9, 6))

    sns.histplot(
        anomaly_scores[y_true == "Normal"],
        color="green",
        label="Normal Traffic",
        stat="density",
        bins=40,
        alpha=0.5
    )

    sns.histplot(
        anomaly_scores[y_true == "Attack"],
        color="red",
        label="Attack Traffic",
        stat="density",
        bins=40,
        alpha=0.5
    )

    plt.axvline(
        0,
        color="black",
        linestyle="--",
        label="Decision Boundary"
    )

    plt.title("Isolation Forest - Anomaly Score Distribution")
    plt.xlabel("Decision Function Score")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "isolation_forest_anomaly_scores.png"
        ),
        dpi=300
    )

    plt.show()

    # Step 9: Save trained model
    print("\n[9] SAVING ISOLATION FOREST MODEL...")

    model_path = os.path.join(
        MODEL_DIR,
        "isolation_forest.pkl"
    )

    joblib.dump(iso_model, model_path)

    print("Model saved successfully:", model_path)

    # Step 10: Return results
    return {
        "model": iso_model,
        "predictions": y_pred,
        "anomaly_scores": anomaly_scores,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }
