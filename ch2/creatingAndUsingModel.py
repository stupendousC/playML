import transformers
from transformers import BertConfig, BertModel, AutoModel
import torch

# AutoModel class, handy when you want to instantiate any model from a checkpoint.
#   The AutoModel class and all of its relatives are actually simple wrappers over the wide variety of models available in the library.
#   It’s a clever wrapper as it can automatically guess the appropriate model architecture for your checkpoint, and then instantiates a model with this architecture.

# =============

# However, if you know the type of model you want to use, you can use the class that defines its architecture directly.
#   Let’s take a look at how this works with a BERT model.

# Building the config
bertConfig = BertConfig()
# Building the model from the config
model1 = BertModel(bertConfig)

print(f"BertConfig() = {bertConfig}")
# => BertConfig {
#   [...]
#   "hidden_size": 768,             <-- noted
#   "intermediate_size": 3072,
#   "max_position_embeddings": 512,
#   "num_attention_heads": 12,
#   "num_hidden_layers": 12,        <-- noted
#   [...]
# }

# What if we used AutoModel instead of BertModel?
autoModel = AutoModel.from_config(bertConfig)
# this gives checkpoint-agnostic code (if your code works for one checkpoint, it should work seamlessly with another.)
# This applies even if the architecture is different, as long as the checkpoint was trained for a similar task (for example, a sentiment analysis task).

print("right now the Model is RANDOMLY INITIALIZED!")
# model1 is usable in this state but will output gibberish unless we train it first.
# but we don't want to train the model from scratch bc req time/data/$, boooo...
# reuse pre-trained models then!
pretrained_model = BertModel.from_pretrained("bert-base-cased")
# this time we don't use BertConfig, instead we loaded a pretrained model via the bert-base-cased checkpoint/identifier.
#   The identifier used to load the model can be the identifier of any model on the Model Hub, as long as it is compatible with the BERT architecture.
#   The entire list of available BERT checkpoints can be found at https://huggingface.co/models?filter=bert
# This model is now initialized with all the weights of the checkpoint.
#   It can be used directly for inference on the tasks it was trained on, and it can also be fine-tuned on a new task.
#   The weights have been downloaded and cached (so future calls to the from_pretrained() method won’t re-download them)

### Saving methods
pretrained_model.save_pretrained("created_by_ch2_creatingAndUsingModel")
# results in these 2 files in the directory:
#   config.json - has metadata & attribs necessary to build the MODEL ARCHITECTURE
#   pytorch_model.bin - state dictionary, contains all the MODEL WEIGHTS/PARAMS.

### Using a Transformer model for inference <- make some predictions!
# as seen in ch2/tokenizerAndModel.py, we used tokenizer(raw_input_strs) to produce a bunch of encoded_sequences,
#   which we run torch.tensor() on, to produce "input_ids" as tensor.

sequences = ["Hello!", "Cool.", "Nice!"]
# pretend a tokenizer gave us these encoded_sequences
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
model_inputs = torch.tensor(encoded_sequences)
print(f"Model inputs = {model_inputs}")

# While the model accepts a lot of different arguments, only the input IDs are necessary.
#   We’ll explain what the other arguments do and when they are required later,
#   but first we need to take a closer look at the tokenizers that build the inputs that a Transformer model can understand.
model_output = pretrained_model(model_inputs)
print(f"Model outputs = {model_output}")