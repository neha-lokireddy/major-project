
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
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
# RANDOM FOREST TRAINING
# ==========================================

def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("       RANDOM FOREST CLASSIFIER")
    print("=" * 60)

    # Step 1: Initialize model
    print("\n[1] INITIALIZING RANDOM FOREST...")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    print("Model initialized successfully.")

    # Step 2: Train model
    print("\n[2] TRAINING RANDOM FOREST...")

    rf_model.fit(X_train, y_train)

    print("Training completed successfully!")

    # Step 3: Generate predictions
    print("\n[3] GENERATING TEST PREDICTIONS...")

    y_pred = rf_model.predict(X_test)

    print("Predictions generated successfully.")

    # Step 4: Evaluate model
    print("\n[4] MODEL EVALUATION")

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nAccuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1-Score :", round(f1 * 100, 2), "%")

    # Step 5: Classification report
    print("\n[5] CLASSIFICATION REPORT")

    labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )

    # Step 6: Confusion matrix
    print("\n[6] GENERATING CONFUSION MATRIX...")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title("Random Forest - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "random_forest_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.show()

    # Step 7: Feature importance
    print("\n[7] FEATURE IMPORTANCE ANALYSIS...")

    feature_importance = rf_model.feature_importances_

    top_indices = np.argsort(feature_importance)[-15:]

    plt.figure(figsize=(10, 7))

    plt.barh(
        range(len(top_indices)),
        feature_importance[top_indices]
    )

    plt.yticks(
        range(len(top_indices)),
        [str(i) for i in top_indices]
    )

    plt.title("Random Forest - Top 15 Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Encoded Feature Index")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "random_forest_feature_importance.png"
        ),
        dpi=300
    )

    plt.show()

    # Step 8: Save trained model
    print("\n[8] SAVING RANDOM FOREST MODEL...")

    model_path = os.path.join(
        MODEL_DIR,
        "random_forest.pkl"
    )

    joblib.dump(rf_model, model_path)

    print("Model saved successfully:", model_path)

    # Step 9: Return results
    return {
        "model": rf_model,
        "predictions": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }
