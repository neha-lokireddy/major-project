
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# ATTACK CATEGORY MAPPING
# ==========================================

ATTACK_MAPPING = {
    "normal": "Normal",

    # Denial of Service
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "worm": "DoS",
    "apache2": "DoS",

    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",

    # Remote to Local
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "snmpgetattack": "R2L",
    "named": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "sendmail": "R2L",
    "httptunnel": "R2L",
    "snmpguess": "R2L",

    # User to Root
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R"
}


# ==========================================
# LABEL PREPARATION
# ==========================================

def prepare_labels(data):

    data = data.copy()

    data["label"] = (
        data["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\.$", "", regex=True)
    )

    data["attack_category"] = data["label"].map(
        ATTACK_MAPPING
    )

    # Check for unmapped attack labels
    unmapped = data.loc[
        data["attack_category"].isna(), "label"
    ].unique()

    if len(unmapped) > 0:
        raise ValueError(
            f"Unmapped attack labels found: {unmapped}"
        )

    # Binary classification target
    data["binary_label"] = np.where(
        data["label"] == "normal",
        "Normal",
        "Attack"
    )

    return data


# ==========================================
# MAIN PREPROCESSING FUNCTION
# ==========================================

def preprocess_data(train_data, test_data):

    print("\n" + "=" * 60)
    print("       DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)

    # Step 1: Prepare labels
    print("\n[1] PREPARING TARGET LABELS...")

    train_data = prepare_labels(train_data)
    test_data = prepare_labels(test_data)

    print("Attack categories identified successfully.")

    print("\nTraining category distribution:")
    print(train_data["attack_category"].value_counts())

    print("\nBinary label distribution:")
    print(train_data["binary_label"].value_counts())

    # Step 2: Separate features and targets
    print("\n[2] SEPARATING FEATURES AND TARGETS...")

    excluded_columns = [
        "label",
        "difficulty",
        "attack_category",
        "binary_label"
    ]

    X_train = train_data.drop(
        columns=excluded_columns
    )

    X_test = test_data.drop(
        columns=excluded_columns
    )

    y_train_multiclass = train_data["attack_category"]
    y_test_multiclass = test_data["attack_category"]

    y_train_binary = train_data["binary_label"]
    y_test_binary = test_data["binary_label"]

    # Step 3: Identify feature types
    print("\n[3] IDENTIFYING NUMERICAL AND CATEGORICAL FEATURES...")

    categorical_features = [
        "protocol_type",
        "service",
        "flag"
    ]

    numerical_features = [
        col for col in X_train.columns
        if col not in categorical_features
    ]

    print("Numerical features:", len(numerical_features))
    print("Categorical features:", len(categorical_features))

    # Step 4: Build preprocessing pipeline
    print("\n[4] ENCODING AND SCALING FEATURES...")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(with_mean=False),
                numerical_features
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Transform test data using training-fitted pipeline
    X_test_processed = preprocessor.transform(
        X_test
    )

    print("Categorical encoding completed.")
    print("Numerical scaling completed.")

    # Step 5: Prepare normal-only training data for anomaly detection
    print("\n[5] PREPARING DATA FOR ANOMALY DETECTION...")

    normal_mask = (
        y_train_binary.to_numpy() == "Normal"
    )

    X_train_normal = X_train_processed[normal_mask]

    print("Normal training records:", X_train_normal.shape[0])

    # Step 6: Save fitted preprocessing pipeline
    print("\n[6] SAVING PREPROCESSING PIPELINE...")

    joblib.dump(
        preprocessor,
        os.path.join(MODEL_DIR, "preprocessor.pkl")
    )

    print("Preprocessor saved successfully!")

    # Step 7: Final summary
    print("\n[7] PREPROCESSING SUMMARY")

    print("Original training shape:", X_train.shape)
    print("Processed training shape:", X_train_processed.shape)

    print("Original testing shape:", X_test.shape)
    print("Processed testing shape:", X_test_processed.shape)

    print("\nMulticlass categories:")
    print(sorted(y_train_multiclass.unique()))

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")

    return (
        X_train_processed,
        X_test_processed,
        y_train_multiclass,
        y_test_multiclass,
        y_train_binary,
        y_test_binary,
        X_train_normal,
        preprocessor
    )
