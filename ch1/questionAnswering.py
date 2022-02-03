from transformers import pipeline

question_answerer = pipeline("question-answering")
results = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

# returns: {'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
print(results)

results = question_answerer(
    question="Would I enjoy a steak dinner?",
    context="I have just gotten bitten by a lone star tick, and now I'm allergic to meat, even though I love steak.",
)
print(results)
# at least it came up w/ score of 0.1 even though it erroneously answered "i love steak" lol