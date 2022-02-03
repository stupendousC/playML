# https://huggingface.co/course/chapter2/6?fw=pt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

### Tokenizer can handle a single sequence or [sequences]
sequence1 = "I've been waiting for a HuggingFace course my whole life."
seq1_model_inputs = tokenizer(sequence1)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
sequences_model_inputs = tokenizer(sequences)

### Tokenizer can pad according to sev objectives:
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")
# Will pad the sequences up to the model max length (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")
# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

### Tokenizer can truncate sequences:
# Will truncate the sequences that are longer than the model max length (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)
# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)

### Tokenizer can convert to specific framework tensors, which can then be sent to the model.
# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")
# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")

### Special tokens, like CLS/SEP/etc
seq1_model_inputs = tokenizer(sequence1)
seq1_input_ids = seq1_model_inputs["input_ids"]
# => [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
seq1_input_ids_decoded = tokenizer.decode(seq1_input_ids)
# => "[CLS] i've been waiting for a huggingface course my whole life. [SEP]"

seq1_tokens = tokenizer.tokenize(sequence1)
seq1_input_ids_dimensionless = tokenizer.convert_tokens_to_ids(seq1_tokens)
# => [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
seq1_input_ids_dimensionless_decoded = tokenizer.decode(seq1_input_ids_dimensionless)
# => "i've been waiting for a huggingface course my whole life."

# the dimensioned seq1_input_ids example had [CLS] & [SEP]
# This is because the model was pretrained with those, so to get the same results for inference we need to add them as well.
# Note that some models don’t add special words, or add different ones;
# models may also add these special words only at the beginning, or only at the end.
# In any case, the tokenizer knows which ones are expected and will deal with this for you.

# As seen in multipleSequences.py, I thought we want to feed the torch.tensor(dimensioned input_ids) to model otherwise it fails.
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model(torch.tensor([seq1_input_ids]))
# => SequenceClassifierOutput(loss=None, logits=tensor([[-1.5607,  1.6123]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
model(torch.tensor([seq1_input_ids_dimensionless]))
# => SequenceClassifierOutput(loss=None, logits=tensor([[-2.7276,  2.8789]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)


### WRAPPING UP: TOKENIZER to MODEL
# going back to teh sequences of diff-length strings (add adding!)
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
# => SequenceClassifierOutput(loss=None, logits=tensor([[-1.5607,  1.6123],
#         [-3.6183,  3.9137]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)

### QUIZ
# Q: What is the order of the language modeling pipeline?
#  The tokenizer handles text and returns IDs.
#  The model handles these IDs and outputs a prediction.
#  The tokenizer can then be used once again to convert these predictions back to some text.

# Q: How many dimensions does the tensor output by the base Transformer model have, and what are they?
# The sequence length, the batch size, and the hidden size

# Q:What is a model head?
#  An additional component, usually made up of one or a few layers, to convert the transformer predictions to a task-specific output

# Q: What is an AutoModel?
# An object that returns the correct architecture based on the checkpoint

# Q: What are the techniques to be aware of when batching sequences of different lengths together?
# Truncating, Padding, and Attention masking

# Q: What is the point of applying a SoftMax function to the logits output by a sequence classification model?
#  It applies a lower and upper bound so that they're understandable. (bound between 0 and 1)
#  AND, The total sum of the output is then 1, resulting in a possible probabilistic interpretation.

# Q:"
