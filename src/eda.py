
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create directory for graphs
os.makedirs("results/graphs", exist_ok=True)

sns.set_theme(style="whitegrid")


# Map individual attacks into their four major categories
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

    # User to Root
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R"
}


def normalize_labels(data):
    data = data.copy()
    data["label"] = (
        data["label"].astype(str).str.strip().str.lower()
        .str.removesuffix(".")
    )
    return data


def run_eda(train_data, test_data):

    print("\n" + "=" * 60)
    print("       NETWORK INTRUSION DETECTION - EDA")
    print("=" * 60)

    train_data = normalize_labels(train_data)
    test_data = normalize_labels(test_data)

    # 1. Dataset information
    print("\n[1] DATASET OVERVIEW")

    print("Training Shape:", train_data.shape)
    print("Testing Shape:", test_data.shape)

    print("\nTraining Dataset Information:")
    train_data.info()

    # 2. Missing values
    print("\n[2] MISSING VALUE ANALYSIS")

    print("Training Missing Values:",
          train_data.isnull().sum().sum())

    print("Testing Missing Values:",
          test_data.isnull().sum().sum())

    # 3. Duplicate records
    print("\n[3] DUPLICATE RECORD ANALYSIS")

    print("Training Duplicates:",
          train_data.duplicated().sum())

    print("Testing Duplicates:",
          test_data.duplicated().sum())

    # 4. Individual attack distribution
    print("\n[4] ATTACK DISTRIBUTION")

    print("\nTraining Data:")
    print(train_data["label"].value_counts())

    print("\nTesting Data:")
    print(test_data["label"].value_counts())

    # 5. Binary classification distribution
    train_data["binary_label"] = (
        train_data["label"].apply(
            lambda x: "Normal" if x == "normal" else "Attack"
        )
    )

    print("\n[5] NORMAL VS ATTACK TRAFFIC")

    print(train_data["binary_label"].value_counts())

    plt.figure(figsize=(7, 5))
    sns.countplot(
        data=train_data,
        x="binary_label",
        order=["Normal", "Attack"]
    )

    plt.title("Normal vs Attack Traffic Distribution")
    plt.xlabel("Traffic Type")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(
        "results/graphs/binary_distribution.png",
        dpi=300
    )
    plt.show()

    # 6. Major attack categories
    train_data["attack_category"] = (
        train_data["label"].map(ATTACK_MAPPING)
        .fillna("Other/Unmapped")
    )

    print("\n[6] MAJOR ATTACK CATEGORIES")

    print(train_data["attack_category"].value_counts())

    print("\nUnmapped attack labels:")
    print(sorted(
        set(train_data["label"])
        - set(ATTACK_MAPPING)
    ))

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=train_data,
        x="attack_category",
        order=["Normal", "DoS", "Probe", "R2L", "U2R",
               "Other/Unmapped"]
    )

    plt.title("Distribution of Major Attack Categories")
    plt.xlabel("Attack Category")
    plt.ylabel("Number of Records")
    plt.tight_layout()

    plt.savefig(
        "results/graphs/attack_categories.png",
        dpi=300
    )
    plt.show()

    # 7. Numerical feature distributions
    print("\n[7] NUMERICAL FEATURE ANALYSIS")

    numerical_features = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "count",
        "srv_count",
        "serror_rate"
    ]

    train_data[numerical_features].hist(
        figsize=(14, 9),
        bins=30
    )

    plt.suptitle("Distribution of Selected Network Features")
    plt.tight_layout()

    plt.savefig(
        "results/graphs/numerical_distributions.png",
        dpi=300
    )
    plt.show()

    # 8. Correlation analysis
    print("\n[8] FEATURE CORRELATION ANALYSIS")

    correlation = train_data[
        numerical_features
    ].corr()

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Between Selected Network Features")
    plt.tight_layout()

    plt.savefig(
        "results/graphs/feature_correlation.png",
        dpi=300
    )
    plt.show()

    print("\nEDA COMPLETED SUCCESSFULLY!")
    print("Graphs saved inside results/graphs/")
