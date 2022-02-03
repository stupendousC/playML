from transformers import pipeline

unmasker = pipeline("fill-mask")
results = unmasker("This course will teach you all about <mask> models.", top_k=2)
# top_k = how many possibilities u want
# model fills in the special <mask> word, aka "mask token"

print(f"fillMask.py... \nresults={results}")

# Can also use the Hosted inference API on the Model Hub!
    # https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France.