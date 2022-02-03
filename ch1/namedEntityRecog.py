from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
results = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

# returns...
# [{'entity_group': 'PER', 'score': 0.99816, 'word': 'Sylvain', 'start': 11, 'end': 18},
#  {'entity_group': 'ORG', 'score': 0.97960, 'word': 'Hugging Face', 'start': 33, 'end': 45},
#  {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Brooklyn', 'start': 49, 'end': 57}
#  ]
# grouped_entities=True allows 'Hugging Face' grouped as a single org
print(results)