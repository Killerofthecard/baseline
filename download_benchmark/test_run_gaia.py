import os

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from openai import OpenAI

load_dotenv()

data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
dataset = load_dataset(data_dir, "2023_level1", split="validation")

example = dataset[0]
question = example["Question"]
print("Question:", question)
print("=" * 80)

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

response = client.chat.completions.create(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ],
)

print(response.choices[0].message.content, example["Final answer"])
