
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result1 = classifier("I've been waiting for a HuggingFace course my whole life")

result2 = classifier(
    ["I've been waiting for a HuggingFace course my whole life.",
     "I hate this so much",
     "I can just die from so much happiness",   # technically positive
     "Of course I'd love to be unvaccinated and end up on life support, that sounds super fun!",  # sarcasm
     ])

print(f"result1 = {result1}")
print(f"\nresult2 = {result2}")


