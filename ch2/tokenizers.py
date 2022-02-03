# https://huggingface.co/course/chapter2/4?fw=pt
# TOKENIZERS translate text into data that can be processed by the model
# and MODELS can only process numbers

# Diff types of tokenizers:
#   WORD-based: split on spaces/punctuation/etc
#       Each word gets an ID (0->vocab.size), which the model uses
#       "unknown"/UNK tokens needed to represent words not in vocab, goal for tokenizer is to minimize these
#   CHARACTER-based: split into indiv chars
#       vocab is smaller :-)
#       fewer unknown tokens :-)
#       large amts of tokens for model processing :-(
#   SUBWORD-based:  Combines 2 approaches above
#       ex: snowboarding -> snow, board, ing
#   BYTE-level BPE
#       as seen in GPT-2
#   WORDPIECE
#       as seen in BERT
#   SENTENCEPIECE/UNIGRAM
#       as seen in sev multilingual models
#   Many more...

import transformers
from transformers import BertTokenizer, AutoTokenizer
# LOAD tokenizer with XyzTokenizer.from_pretrained(abc-checkpoint)
#       note that loading models are done the same way!
#           XyzModel.from_pretrained(abc-checkpoint)
#           as seen in creatingAndUsingModel.py!
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# => PreTrainedTokenizer(name_or_path='bert-base-cased', vocab_size=28996, model_max_len=512, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})
tokenizer2 = AutoTokenizer.from_pretrained("bert-base-cased")
# => PreTrainedTokenizerFast(name_or_path='bert-base-cased', vocab_size=28996, model_max_len=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})

tokenz_results = tokenizer("Using a Transformer network is simple")
print(f"tokenizer(someString) results = {tokenz_results}")
# => {'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

# SAVE tokenizer with XyzTokenizer.save_pretrained("directoryName")
#       note that saving models are done the same way!
#           XyzModel.save_pretrained("directoryName")
tokenizer.save_pretrained("created_by_ch3_tokenizers")

### ENCODING = (text -> numbers)
#       step 1: text -> tokens          ... Need to instant tokenizer using name of model to ensure same rules
#       step 2: tokens -> id_numbers    ... Tokenizer uses same vocab from the pretrained model
#       step 3: tensor(id_numbers)      ... gives the Tensors that models need

sequence = "Using a Transformer network is simple"
# can use tokenizer or tokenizer2 from above

tokens = tokenizer.tokenize(sequence)
print(tokens)
# => ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
# => [7993, 170, 11303, 1200, 2443, 1110, 3014]

### DECODING (vocab indices -> text)
decoded_string = tokenizer.decode(ids)
print(decoded_string)
# => 'Using a Transformer network is simple'