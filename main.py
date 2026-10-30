
import os
import pandas as pd

from src.eda import run_eda
from src.preprocessing import preprocess_data
from src.supervised_model import train_random_forest
from src.anomaly_detection import train_isolation_forest


# ==========================================
# DATASET CONFIGURATION
# ==========================================

TRAIN_PATH = "data/raw/KDDTrain+.txt"
TEST_PATH = "data/raw/KDDTest+.txt"


# ==========================================
# DATASET COLUMN NAMES
# ==========================================

columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    print("\n" + "=" * 60)
    print("       NETWORK INTRUSION DETECTION SYSTEM")
    print("=" * 60)

    print("\n[1] LOADING NETWORK DATASET...")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_PATH}"
        )

    train_data = pd.read_csv(
        TRAIN_PATH,
        names=columns
    )

    test_data = pd.read_csv(
        TEST_PATH,
        names=columns
    )

    print("\nDataset loaded successfully!")

    print("\nTraining Dataset Shape:", train_data.shape)
    print("Testing Dataset Shape:", test_data.shape)

    print("\nFirst Five Training Records:")
    print(train_data.head())

    print("\nTraining Attack Distribution:")
    print(train_data["label"].value_counts())

    return train_data, test_data


# ==========================================
# MAIN EXECUTION
# ==========================================




def main():

    try:

        # ==========================================
        # STEP 1: LOAD DATASET
        # ==========================================

        train_data, test_data = load_dataset()


        # ==========================================
        # STEP 2: EXPLORATORY DATA ANALYSIS
        # ==========================================

        print("\n[2] STARTING EXPLORATORY DATA ANALYSIS...")

        run_eda(train_data, test_data)


        # ==========================================
        # STEP 3: DATA PREPROCESSING
        # ==========================================

        print("\n[3] STARTING DATA PREPROCESSING...")

        (
            X_train,
            X_test,
            y_train_multiclass,
            y_test_multiclass,
            y_train_binary,
            y_test_binary,
            X_train_normal,
            preprocessor
        ) = preprocess_data(train_data, test_data)


        # ==========================================
        # STEP 4: DATA PREPARATION SUMMARY
        # ==========================================

        print("\n" + "=" * 60)
        print("       NIDS DATA PREPARATION SUMMARY")
        print("=" * 60)

        print("\nTraining Features:", X_train.shape)
        print("Testing Features:", X_test.shape)

        print("\nMulticlass Training Labels:")
        print(y_train_multiclass.value_counts())

        print("\nBinary Training Labels:")
        print(y_train_binary.value_counts())

        print("\nNormal Samples for Isolation Forest:",
              X_train_normal.shape[0])

        print("\nPreprocessing Pipeline: SAVED")


        # ==========================================
        # STEP 5: RANDOM FOREST TRAINING
        # ==========================================

        print("\n[5] STARTING RANDOM FOREST TRAINING...")

        rf_results = train_random_forest(
            X_train,
            y_train_multiclass,
            X_test,
            y_test_multiclass
        )


        # ==========================================
        # STEP 6: RANDOM FOREST RESULTS SUMMARY
        # ==========================================

        print("\n" + "=" * 60)
        print("       RANDOM FOREST RESULTS SUMMARY")
        print("=" * 60)

        print(f"\nAccuracy: {rf_results['accuracy'] * 100:.2f}%")
        print(f"Precision: {rf_results['precision'] * 100:.2f}%")
        print(f"Recall: {rf_results['recall'] * 100:.2f}%")
        print(f"F1-Score: {rf_results['f1_score'] * 100:.2f}%")

        print("\nTrained Model: models/random_forest.pkl")


        # ==========================================
        # STEP 7: ISOLATION FOREST TRAINING
        # ==========================================

        print("\n[7] STARTING ISOLATION FOREST TRAINING...")

        iso_results = train_isolation_forest(
            X_train_normal,
            X_test,
            y_test_binary
        )


        # ==========================================
        # STEP 8: ISOLATION FOREST RESULTS SUMMARY
        # ==========================================

        print("\n" + "=" * 60)
        print("       ISOLATION FOREST RESULTS SUMMARY")
        print("=" * 60)

        print(f"\nAccuracy: {iso_results['accuracy'] * 100:.2f}%")
        print(f"Precision: {iso_results['precision'] * 100:.2f}%")
        print(f"Recall: {iso_results['recall'] * 100:.2f}%")
        print(f"F1-Score: {iso_results['f1_score'] * 100:.2f}%")

        print("\nTrained Model: models/isolation_forest.pkl")


        # ==========================================
        # STEP 9: FINAL PIPELINE SUMMARY
        # ==========================================

        print("\n" + "=" * 60)
        print("       NIDS BASELINE PIPELINE COMPLETED")
        print("=" * 60)

        print("\nCompleted Stages:")

        print("1. Dataset Loading")
        print("2. Exploratory Data Analysis")
        print("3. Data Preprocessing")
        print("4. Random Forest Classification")
        print("5. Isolation Forest Anomaly Detection")

        print("\nSaved Models:")
        print("- models/preprocessor.pkl")
        print("- models/random_forest.pkl")
        print("- models/isolation_forest.pkl")

        print("\nNext Stage: Hybrid Decision Engine")

        print("\n" + "=" * 60)


    except FileNotFoundError as error:

        print("\nERROR:", error)

    except Exception as error:

        print("\nUNEXPECTED ERROR:", error)


if __name__ == "__main__":
    main()
