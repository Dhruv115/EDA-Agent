import pandas as pd
import numpy as np
import logging
import os
from dataclasses import dataclass, field
from groq import Groq

import config

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("EDA-Pipeline")

# ─── Groq client ──────────────────────────────────────────────────────────────
client = Groq(api_key=config.GROQ_API_KEY)

def ask_llm(prompt: str) -> str:
    """Send a prompt to Groq and return the response text."""
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ─── Shared report structure ──────────────────────────────────────────────────
@dataclass
class CleaningReport:
    actions: list[dict] = field(default_factory=list)

    def log(self, agent: str, column: str, action: str, detail: str = ""):
        self.actions.append({
            "agent":  agent,
            "column": column,
            "action": action,
            "detail": detail
        })

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.actions)


# ─── Sub-agents ───────────────────────────────────────────────────────────────

class MissingValueAgent:
    def __init__(self):
        self.drop_col_thresh = config.MISSING_DROP_COL_THRESH
        self.drop_row_thresh = config.MISSING_DROP_ROW_THRESH

    def run(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        df = df.copy()
        log.info("MissingValueAgent started")

        for col in list(df.columns):
            missing_rate = df[col].isna().mean()
            if missing_rate > self.drop_col_thresh:
                df.drop(columns=[col], inplace=True)
                report.log("MissingValueAgent", col, "dropped column", f"{missing_rate:.0%} missing")
                continue
            if missing_rate == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                report.log("MissingValueAgent", col, "imputed with median", f"value={fill_val:.3g}")
            else:
                fill_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
                report.log("MissingValueAgent", col, "imputed with mode", f"value={fill_val}")

        row_missing = df.isna().mean(axis=1)
        bad_rows = row_missing[row_missing > self.drop_row_thresh].index
        if len(bad_rows):
            df.drop(index=bad_rows, inplace=True)
            df.reset_index(drop=True, inplace=True)
            report.log("MissingValueAgent", "rows", "dropped", f"{len(bad_rows)} near-empty rows")

        return df


class DuplicateAgent:
    def run(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        log.info(f"DuplicateAgent: removed {removed} duplicates")
        if removed:
            report.log("DuplicateAgent", "all", "removed duplicates", f"{removed} rows dropped")
        return df


class OutlierAgent:
    def __init__(self):
        self.method = config.OUTLIER_METHOD

    def run(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        df = df.copy()
        log.info(f"OutlierAgent started using method={self.method}")
        num_cols = df.select_dtypes(include="number").columns

        for col in num_cols:
            if self.method == "iqr":
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            else:
                mean, std = df[col].mean(), df[col].std()
                lower = mean - 3 * std
                upper = mean + 3 * std

            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers:
                df[col] = df[col].clip(lower, upper)
                report.log("OutlierAgent", col, "capped outliers",
                           f"{outliers} values → [{lower:.3g}, {upper:.3g}]")

        return df


class DataTypeAgent:
    def run(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        df = df.copy()
        log.info("DataTypeAgent started")

        for col in df.columns:
            original_dtype = str(df[col].dtype)
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().mean() > 0.9:
                    df[col] = converted
                    report.log("DataTypeAgent", col, "cast to numeric", f"was {original_dtype}")
                    continue
                try:
                    parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="raise")
                    df[col] = parsed
                    report.log("DataTypeAgent", col, "cast to datetime", f"was {original_dtype}")
                    continue
                except Exception:
                    pass
                if df[col].nunique() / len(df) < 0.05:
                    df[col] = df[col].astype("category")
                    report.log("DataTypeAgent", col, "cast to category",
                               f"{df[col].nunique()} unique values")

        return df


class FormattingAgent:
    def run(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        df = df.copy()
        log.info("FormattingAgent started")
        str_cols = df.select_dtypes(include="object").columns

        for col in str_cols:
            before = df[col].copy()
            df[col] = df[col].str.strip()
            if config.LOWERCASE_STRINGS:
                df[col] = df[col].str.lower()
            changed = (df[col] != before).sum()
            if changed:
                report.log("FormattingAgent", col, "standardized strings", f"{changed} values changed")

        return df


# ─── LLM Insight Agent ────────────────────────────────────────────────────────

class LLMInsightAgent:
    """Sends the cleaning report to Groq and gets a plain-English summary."""

    def run(self, df: pd.DataFrame, report: CleaningReport) -> str:
        log.info("LLMInsightAgent: asking Groq for insights")

        summary = report.summary().to_string(index=False)
        shape   = df.shape
        dtypes  = df.dtypes.value_counts().to_string()
        nulls   = df.isna().sum().sum()

        prompt = f"""
You are a data analyst assistant. A data cleaning pipeline just ran on a dataset.

Dataset after cleaning:
- Shape: {shape}
- Null values remaining: {nulls}
- Column dtypes:
{dtypes}

Cleaning actions taken:
{summary}

Give a short, clear summary (5-8 bullet points) of:
1. What was wrong with the original data
2. What was fixed and how
3. Any remaining concerns the analyst should check before EDA
"""
        return ask_llm(prompt)


# ─── Aggregator ───────────────────────────────────────────────────────────────

class AggregatorAgent:
    def __init__(self):
        self.agents = [
            MissingValueAgent(),
            DuplicateAgent(),
            OutlierAgent(),
            DataTypeAgent(),
            FormattingAgent(),
        ]

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        report = CleaningReport()
        for agent in self.agents:
            df = agent.run(df, report)
        return df, report


# ─── Controller ───────────────────────────────────────────────────────────────

class DataCleaningPipeline:
    def __init__(self):
        self.aggregator   = AggregatorAgent()
        self.llm_agent    = LLMInsightAgent()

    def profile(self, df: pd.DataFrame):
        log.info("Profiling dataset")
        print("\n=== Dataset Profile ===")
        print(f"Shape      : {df.shape}")
        print(f"Duplicates : {df.duplicated().sum()}")
        missing = df.isna().sum()
        print(f"Missing    :\n{missing[missing > 0]}")
        print(f"Dtypes     :\n{df.dtypes.value_counts()}\n")

    def save_outputs(self, clean_df: pd.DataFrame, report: CleaningReport):
        os.makedirs(os.path.dirname(config.OUTPUT_DATA_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(config.REPORT_OUTPUT_PATH), exist_ok=True)
        clean_df.to_csv(config.OUTPUT_DATA_PATH, index=False)
        report.summary().to_csv(config.REPORT_OUTPUT_PATH, index=False)
        log.info(f"Saved cleaned data → {config.OUTPUT_DATA_PATH}")
        log.info(f"Saved report       → {config.REPORT_OUTPUT_PATH}")

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport, str]:
        self.profile(df)

        clean_df, report = self.aggregator.run(df)

        print("\n=== Cleaning Report ===")
        print(report.summary().to_string(index=False))
        print(f"\nFinal shape: {clean_df.shape}")

        insights = self.llm_agent.run(clean_df, report)
        print("\n=== LLM Insights ===")
        print(insights)

        self.save_outputs(clean_df, report)

        return clean_df, report, insights


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(config.INPUT_DATA_PATH)
    pipeline = DataCleaningPipeline()
    clean_df, report, insights = pipeline.run(df)