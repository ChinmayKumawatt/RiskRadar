from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "informational_plots"


DISEASE_CONFIGS = {
    "heart": {
        "dataset_path": DATA_DIR / "heart.csv",
        "target": "target",
        "features": [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "thalach",
            "exang",
            "thal",
        ],
        "title": "Heart Disease",
    },
    "diabetes": {
        "dataset_path": DATA_DIR / "diabetes.csv",
        "target": "Outcome",
        "features": [
            "Glucose",
            "BloodPressure",
            "BMI",
            "Age",
        ],
        "title": "Diabetes",
    },
    "ckd": {
        "dataset_path": DATA_DIR / "kidney.csv",
        "target": "classification",
        "features": [
            "age",
            "bp",
            "htn",
            "dm",
            "appet",
        ],
        "title": "Chronic Kidney Disease",
    },
    "hypertension": {
        "dataset_path": DATA_DIR / "hypertension.csv",
        "target": "Risk",
        "features": [
            "male",
            "age",
            "cigsPerDay",
            "BPMeds",
            "totChol",
            "BMI",
            "heartRate",
        ],
        "title": "Hypertension",
    },
}


def normalize_dataframe(df, features):
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.apply(
        lambda column: column.str.strip() if column.dtype == "object" else column
    )
    cleaned_df.replace("?", pd.NA, inplace=True)

    category_mapping = {
        "yes": 1,
        "no": 0,
        "good": 1,
        "poor": 0,
        "ckd": 1,
        "notckd": 0,
        "ckd\t": 1,
    }

    for feature in features:
        if cleaned_df[feature].dtype == "object":
            lowered = cleaned_df[feature].astype("string").str.lower()
            cleaned_df[feature] = lowered.map(lambda value: category_mapping.get(value, value))

        cleaned_df[feature] = pd.to_numeric(cleaned_df[feature], errors="coerce")

    return cleaned_df


def save_target_distribution(df, target_column, title, output_path):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_column, data=df, color="#5e9f95")
    plt.title(f"{title} Target Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_numeric_distributions(df, features, title, output_path):
    plot_features = [feature for feature in features if df[feature].notna().any()]
    num_features = len(plot_features)

    cols = 2
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = axes.flatten() if num_features > 1 else [axes]

    for idx, feature in enumerate(plot_features):
        sns.histplot(df[feature].dropna(), kde=True, ax=axes[idx], color="#b55233")
        axes[idx].set_title(feature)

    for idx in range(num_features, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"{title} Feature Distributions", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df, features, title, output_path):
    corr = df[features].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="RdYlBu_r", fmt=".2f", square=True)
    plt.title(f"{title} Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_feature_boxplots(df, features, target_column, title, output_path):
    plot_features = [feature for feature in features if df[feature].notna().any()]
    num_features = len(plot_features)

    cols = 2
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = axes.flatten() if num_features > 1 else [axes]

    for idx, feature in enumerate(plot_features):
        sns.boxplot(x=target_column, y=feature, data=df, ax=axes[idx], color="#cfdcae")
        axes[idx].set_title(feature)

    for idx in range(num_features, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"{title} Feature Spread by Target", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_markdown(df, target_column, features, title, output_path):
    lines = [
        f"# {title} Informational Summary",
        "",
        f"- Rows: {len(df)}",
        f"- Features used: {', '.join(features)}",
        f"- Target column: {target_column}",
        "",
        "## Target counts",
    ]

    target_counts = df[target_column].value_counts(dropna=False).to_dict()
    for label, count in target_counts.items():
        lines.append(f"- {label}: {count}")

    lines.extend(["", "## Feature summary"])
    summary = df[features].describe(include="all").transpose().fillna("")

    lines.append("")
    lines.append("```text")
    lines.append(summary.to_string())
    lines.append("```")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_plots():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for disease_name, config in DISEASE_CONFIGS.items():
        disease_dir = OUTPUT_DIR / disease_name
        disease_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(config["dataset_path"])
        df = normalize_dataframe(df, config["features"])

        save_target_distribution(
            df,
            config["target"],
            config["title"],
            disease_dir / "target_distribution.png",
        )
        save_numeric_distributions(
            df,
            config["features"],
            config["title"],
            disease_dir / "feature_distributions.png",
        )
        save_correlation_heatmap(
            df,
            config["features"],
            config["title"],
            disease_dir / "correlation_heatmap.png",
        )
        save_feature_boxplots(
            df,
            config["features"],
            config["target"],
            config["title"],
            disease_dir / "feature_boxplots_by_target.png",
        )
        save_summary_markdown(
            df,
            config["target"],
            config["features"],
            config["title"],
            disease_dir / "summary.md",
        )

        print(f"Generated informational plots for {disease_name} -> {disease_dir}")


if __name__ == "__main__":
    generate_plots()
