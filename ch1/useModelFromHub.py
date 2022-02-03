from transformers import pipeline

# This is from https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
# Example wanted task=TextGeneration, so we see distilgpt2 model
generator = pipeline("text-generation", model="distilgpt2")
results = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

print(f"useModelFromHub.py... \nresults={results}")