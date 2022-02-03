from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result1 = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(f"Zero shot results = {result1}")