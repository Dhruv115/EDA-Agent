from dotenv import load_dotenv
import os

load_dotenv()

# Secrets from .env
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
LLM_MODEL           = os.getenv("LLM_MODEL", "llama3-70b-8192")
INPUT_DATA_PATH     = os.getenv("INPUT_DATA_PATH", "data/raw/input.csv")
OUTPUT_DATA_PATH    = os.getenv("OUTPUT_DATA_PATH", "data/cleaned/output.csv")
REPORT_OUTPUT_PATH  = os.getenv("REPORT_OUTPUT_PATH", "reports/cleaning_report.csv")
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")

# Pipeline config 
MISSING_DROP_COL_THRESH = 0.6
MISSING_DROP_ROW_THRESH = 0.8
OUTLIER_METHOD          = "iqr"
LOWERCASE_STRINGS       = True

# Validate on startup
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")