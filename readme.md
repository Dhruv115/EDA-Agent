# EDA Agent — AI Powered Data Cleaning Pipeline

AI-powered multi-agent pipeline that automates data cleaning for EDA — handles missing values, duplicates, outliers, type casting, and string formatting, with LLM-generated insights via Groq.

## Agents
| Agent | What it does |
|---|---|
| `MissingValueAgent` | Imputes numerics with median, categoricals with mode. Drops columns >60% missing |
| `DuplicateAgent` | Removes exact duplicate rows |
| `OutlierAgent` | Caps outliers using IQR fencing |
| `DataTypeAgent` | Auto-casts columns to numeric, datetime, or category |
| `FormattingAgent` | Strips whitespace, lowercases strings |
| `LLMInsightAgent` | Sends cleaning report to Groq and returns plain-English summary |

## Setup

1. Clone the repo
```
   git clone https://github.com/Dhruv115/EDA-Agent.git
   cd EDA-Agent
```

2. Install dependencies
```
   pip install pandas numpy groq python-dotenv
```

3. Copy `.env.example` to `.env` and add your Groq API key
```
   cp .env.example .env
```

4. Drop your CSV into `data/` and update `INPUT_DATA_PATH` in `.env`

5. Run the pipeline
```
   python pipeline.py
```

## Output
- Cleaned CSV saved to `data/cleaned/`
- Cleaning report saved to `data/reports/`
- LLM insights printed to terminal

## Tech Stack
Python, Pandas, NumPy, Groq API, python-dotenv