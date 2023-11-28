import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import random
import re
import string

import keras
import numpy as np
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
from keras import layers, ops
from keras.layers import TextVectorization

# Download the data
text_file = keras.utils.get_file(
  fname="spa-eng.zip",
  origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
  extract=True,
)
text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"

# Parse the data
with open(text_file) as f:
  lines = f.read().split("\n")[:-1]

text_pairs = []
for line in lines:
  eng, spa = line.split("\t")
  spa = "[start]" + spa + "[end]"
  text_pairs.append((eng, spa))

for _ in range(5):
  print(random.choice(text_pairs))

# Split data into train, validation, and test
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples : ]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

# Vectorize the data
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64

def custom_standardization(input_string):
  lowercase = tf_strings.lower(input_string)
  return tf_strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

eng_vectorization = TextVectorization(
  max_tokens=vocab_size,
  output_mode="int",
  output_sequence_length=sequence_length,
)

spa_vectorization = TextVectorization(
  max_tokens=vocab_size,
  output_mode="int",
  output_sequence_length=sequence_length+1,
  standardize=custom_standardization,
)

train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)

# Format the data
def format_dataset(eng, spa):
  eng = eng_vectorization(eng)
  spa = spa_vectorization(spa)
  return (
    {
      "encoder_inputs": eng,
      "decoder_inputs": spa[:, :-1],
    },
    spa[:, 1:],
  )

def make_dataset(pairs):
  eng_texts, spa_texts = zip(*pairs)
  eng_texts = list(eng_texts)
  spa_texts = list(spa_texts)
  dataset = tf_data.Dataset.from_tensor_slices((eng_texts, spa_texts))
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(format_dataset)
  return dataset.cache().shuffle(2048).prefetch(16)

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
  print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
  print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
  print(f"targets.shape: {targets.shape}")


