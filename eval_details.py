import pandas as pd
from datasets import load_dataset

# Path to the file you found in evaluations/details/...
file_path = "./evaluations/details/openrouter/z-ai/glm-5/2026-03-09T21-12-18.305146/details_armenian:exam_math|0_2026-03-09T21-12-18.305146.parquet"

df = pd.read_parquet(file_path)

# This lets you see the prompt, the model's answer, and the gold label
pd.set_option("display.max_colwidth", None)
print(df.head(10))

# If you want to look at it in Excel later, just export it:
df.to_csv(file_path.replace(".parquet", ".csv"), index=False)
