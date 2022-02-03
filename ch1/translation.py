from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result = translator("Ce cours est produit par Hugging Face.")

print(result)

result = translator("Je parle francais tres mal!")
print(result)

# ValueError: This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed in order to use this tokenizer.
# added sentnecepiece to requirements.txt, then pip install that in SSH'd EC2 instance, then run this file