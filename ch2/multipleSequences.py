import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)       # this causes exception later
print(f"input_ids = {input_ids}")
# => tensor([ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,
#          2026,  2878,  2166,  1012])

try:
    output_will_fail = model(input_ids)
except Exception:
    print("get... IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)")
    # The problem is that we sent a single sequence to the model, whereas 🤗 Transformers models expect multiple sentences by default.
    # Here we tried to do everything the tokenizer did behind the scenes when we applied it to a sequence,
    # but if you look closely, you’ll see that it didn’t just convert the list of input IDs into a tensor, it added a dimension on top of it:
    tokenized_inputs = tokenizer(sequence, return_tensors="pt")
    print(tokenized_inputs["input_ids"])
    # => tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
    #           2607,  2026,  2878,  2166,  1012,   102]])
    #           ^ dimensionless, so extra numbers in front & behind what's seen in input_ids above


### Try again and add a new dimension
#   input_ids_w_dimension = torch.tensor(batch_of_seq_ids)
#                           where batch_of_seq_ids = [seq1_ids, seq2_ids, etc...]
input_ids_w_dimension = torch.tensor([ids])     # note the diff compared to L12!
print("Input IDs w dimension:", input_ids_w_dimension)
# =>  [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]

output = model(input_ids_w_dimension)
print("Logits:", output.logits)
# => [[-2.7276,  2.8789]]

### PADDING THE INPUTS
#   Problem arises when batch_of_seq_ids were from diff-length text!
#       bad_shape_batched_ids = [   [200, 200, 200],
#                                   [200, 200]            ]
#   Bc Tensor needs to be rectangle shape.  So we'll need to PAD IT!
#   For example, if you have 10 sentences with 10 words and 1 sentence with 20 words,
#       padding will ensure all the sentences have 20 words.
#   btw The padding token ID can be found in tokenizer.pad_token_id
#   In our example, the resulting tensor looks like this:
random_padding_id = 100
padding_id = tokenizer.pad_token_id     # happens to be 0 here
    # idk what the importance of the actual padding_id value is.
    # Prob best to stick to the official tokenizer.pad_token_id, instead of making up a random number

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    sequence1_ids[0],
    sequence2_ids[0] + [padding_id],
]

seq1_logits = model(torch.tensor(sequence1_ids)).logits
# => tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
seq2_logits = model(torch.tensor(sequence2_ids)).logits
# => tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
batched_logits = model(torch.tensor(batched_ids)).logits
# => tensor([[ 1.5694, -1.3895], [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)
# NOTE the batched_logits' 2nd logits above does NOT match up to seq2_logits!!!
# This is because the key feature of Transformer models is attention layers that contextualize each token.
#   also, seq2_logits was computed w/o the padding_id
# To get the same result when passing individual sentences of different lengths through the model
#   or when passing a batch with the same sentences and padding applied,
#   we need to tell those attention layers to IGNORE the padding tokens.
#   This is done by using an ATTENTION MASK.

### ATTENTION MASKS
# Attention masks are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s:
#   1s indicate the corresponding tokens should be attended to, and
#   0s indicate the corresponding tokens should not be attended to
#       (i.e., they should be ignored by the attention layers of the model).
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]
outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
# => tensor([[ 1.5694, -1.3895],
#         [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)     <-- yup!!! same as seq2_logits!


### LONGER SEQUENCES
# With Transformer models, there is a limit to the lengths of the sequences we can pass the models.
# Most models handle sequences of up to 512 or 1024 tokens, and will crash when asked to process longer sequences.
# There are two solutions to this problem:
#   Use a model with a longer supported sequence length.
#   Truncate your sequences:    sequence = sequence[:max_sequence_length]
# Models such as Longformer & LED specialize in very long seqs.