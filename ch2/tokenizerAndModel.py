# from https://huggingface.co/course/chapter2/2?fw=pt
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

#  default checkpoint of the "sentiment-analysis: pipeline is distilbert-base-uncased-finetuned-sst-2-english
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Once we have the tokenizer, we can directly pass our sentences to it and we’ll get back a dictionary that’s ready to feed to our model!
# #The only thing left to do is to convert the list of input IDs to tensors.

# You can use 🤗 Transformers without having to worry about which ML framework is used as a backend;
# it might be PyTorch or TensorFlow, or Flax for some models.
# However, Transformer models only accept tensors as input.
# If this is your first time hearing about tensors, you can think of them as NumPy arrays instead.
# A NumPy array can be a scalar (0D), a vector (1D), a matrix (2D), or have more dimensions.
# It’s effectively a tensor; other ML frameworks’ tensors behave similarly, and are usually as simple to instantiate as NumPy arrays.

# To specify the type of tensors we want to get back (PyTorch, TensorFlow, or plain NumPy), we use the return_tensors argument:
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(f"tokenizer(raw_inputs, ...) = \n{inputs}")
# inputs = {
#     'input_ids': tensor([
#         [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
#         [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
#     ]),
#     'attention_mask': tensor([
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     ])
# }

# Transformers provides an AutoModel class which also has a from_pretrained() method:
model = AutoModel.from_pretrained(checkpoint)

# given some inputs, it outputs what we’ll call hidden states, also known as features.
# For each model input, we’ll retrieve a high-dimensional vector representing the contextual understanding of that input by the Transformer model.
# The vector output has 3 dimensions:
#     batch size = # of seq processed at a time (in above we have 2 in raw_inputs)
#     sequence length = The length of the numerical representation of the sequence (16 in our example inputs['input_ids']).
#     hidden size = The vector dimension of each model input (768 for smaller models,else can get to 3072+)

outputs = model(**inputs)
print(f"hidden state.shape = {outputs.last_hidden_state.shape}")
# these intermediate hidden states are auto-processed by the model head

print(f"what is outputs?? {outputs}")
# but it does NOT have an outputs.logits which is the what model Head would output
# This AutoModel class doesn't have a head, we need one from AutoModelForSequenceClassification
#   There are a bunch of other AutoModelForXYZ dpeending on the task.

# ================
model_with_head = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model_with_head(**inputs)

print("logit shape = ", outputs.logits.shape)
# => torch.Size([2, 2])
# Since we have just two sentences and two labels, the result we get from our model is of shape 2 x 2.

# this is the actual model output at the end, but it's raw & unnormalized
print("outputs.logits = ", outputs.logits)
# => tensor([[-1.5607,  1.6123],
#         [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)

# now turn them into actual probabilities
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print("predictions = ", predictions)
# => tensor([[4.0195e-02, 9.5980e-01],
#         [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
# Now we can see that the model predicted [0.0402, 0.9598] for the first sentence
# and [0.9995, 0.0005] for the second one. These are recognizable probability scores.

# To get the labels corresponding to each position,
print(f"labels = {model.config.id2label}")
# => {0: 'NEGATIVE', 1: 'POSITIVE'}

# We have successfully reproduced the three steps of the pipeline:
#   preprocessing with tokenizers,
#   passing the inputs through the model, and
#   postprocessing!

