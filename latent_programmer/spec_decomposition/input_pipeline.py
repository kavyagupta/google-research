# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input pipeline for Robust-fill dataset."""

import tensorflow.compat.v2 as tf

from latent_programmer.tasks.robust_fill import dsl as robust_fill_dsl
from latent_programmer.tasks.scan import scan_vocab

gfile = tf.io.gfile

SEPARATOR_TOKEN = '|'


def create_robust_fill_dataset(
    file_pattern, spec_token_id_table, num_examples, renaming_dict):
  """Loads a RobustFill step-by-step dataset.

  Args:
    file_pattern: A file pattern for the TFRecord files to read.
    spec_token_id_table: Mapping from characters (tokens) to token IDs for the
      I/O specification vocabulary.
    num_examples: The number of examples in an I/O specification.
    renaming_dict: A dict mapping from the new name of fields in this dataset to
      the old name as in the original TFRecord files.

  Returns:
    A tf.data.Dataset containing dictionaries where the keys are the same as in
    `renaming_dict`.
  """
  filenames = gfile.glob(file_pattern)
  raw_dataset = tf.data.TFRecordDataset(filenames)

  spec_vocab_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(
          # Add padding.
          [''] + list(spec_token_id_table.keys()),
          [0] + list(spec_token_id_table.values()),
          key_dtype=tf.string,
          value_dtype=tf.int64),
      len(spec_token_id_table) + 1)
  eos_id = spec_token_id_table[robust_fill_dsl.EOS]

  def _parse_fn(record):
    """Parses a record into a feature_dict."""
    empty_default = [''] * num_examples
    feature_values = tf.io.parse_single_example(
        serialized=record,
        features={
            'inputs':
                tf.io.FixedLenFeature([num_examples], tf.string,
                                      default_value=empty_default),
            'outputs':
                tf.io.FixedLenFeature([num_examples], tf.string,
                                      default_value=empty_default),
            'next_part':
                tf.io.FixedLenFeature([num_examples], tf.string,
                                      default_value=empty_default),
            'program_part':
                tf.io.FixedLenFeature([], tf.string, default_value=''),
        })

    # Map characters to tokens.
    inputs = tf.strings.unicode_split(
        feature_values['inputs'], 'UTF-8').to_tensor()
    inputs = spec_vocab_table.lookup(inputs)

    outputs = tf.strings.unicode_split(
        feature_values['outputs'], 'UTF-8').to_tensor()
    outputs = spec_vocab_table.lookup(outputs)

    next_part = tf.strings.unicode_split(
        feature_values['next_part'], 'UTF-8').to_tensor()
    next_part = spec_vocab_table.lookup(next_part)

    joined_next_part = tf.strings.reduce_join(feature_values['next_part'],
                                              separator=SEPARATOR_TOKEN)
    joined_next_part = tf.strings.unicode_split(joined_next_part, 'UTF-8')
    joined_next_part = spec_vocab_table.lookup(joined_next_part)
    joined_next_part = tf.concat([joined_next_part, [eos_id]], axis=-1)

    program_part = tf.strings.split(feature_values['program_part'], sep=' ')
    program_part = tf.strings.to_number(program_part, out_type=tf.int32)
    program_part = tf.concat([program_part, [eos_id]], axis=-1)

    # inputs: [num_strings, max_length_of_input]
    # outputs: [num_strings, max_length_of_output]
    # next_part: [num_strings, max_length_of_output_part]
    # joined_next_part: [num_strings * (max_length_of_part + 1)]
    # program_part: [max_length_of_program_part]
    all_data_dict = {
        'inputs': inputs,
        'outputs': outputs,
        'next_part': next_part,
        'joined_next_part': joined_next_part,
        'program_part': program_part,
    }

    return {
        new_name: all_data_dict[old_name]
        for new_name, old_name in renaming_dict.items()
    }

  dataset = raw_dataset.map(_parse_fn)
  return dataset


def create_scan_dataset_from_tf_record(
    file_pattern, token_id_table, char_id_table, use_bos_separators):
  """Returns an instance of tf.data.Dataset."""
  del char_id_table
  if 1 + 1 == 2:  # Avoid an unreachable code lint error.
    raise NotImplementedError()  # TODO(kshi): Implement.

  filenames = gfile.glob(file_pattern)
  raw_dataset = tf.data.TFRecordDataset(filenames)

  bos_token = token_id_table[scan_vocab.BOS]
  eos_token = token_id_table[scan_vocab.EOS]

  def _parse_fn(record):
    """Parses a record into a feature_dict."""
    feature_values = tf.io.parse_single_example(
        serialized=record,
        features={
            'input': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'output': tf.io.FixedLenFeature([], tf.string, default_value=''),
        })

    # Step 1. Parse input tokens.
    input_str = feature_values['input']
    input_split = tf.strings.split(input_str, sep=' ')
    input_tokens = tf.strings.to_number(input_split, out_type=tf.int32)
    input_tokens = tf.expand_dims(input_tokens, axis=0)

    # Step 2. Create dummy "output" (analogous to RobustFill's output).
    dummy = input_tokens

    # Step 3. Parse output program into tokens.
    program = feature_values['output']
    # Add BOS between every part, then add BOS followed by EOS. `program` has a
    # | between parts (not at the beginning or end of the sequence, and no
    # spaces around |).
    if use_bos_separators:
      program = tf.strings.join([
          tf.strings.regex_replace(program, r'\|', ' {} '.format(bos_token)),
          ' {} {}'.format(bos_token, eos_token),
      ])
    else:
      program = tf.strings.join([
          tf.strings.regex_replace(program, r'\|', ' '),
          ' {}'.format(eos_token),
      ])
    # Parse numbers.
    program = tf.strings.split(program, sep=' ')
    program = tf.strings.to_number(program, out_type=tf.int32)

    return input_tokens, dummy, program

  dataset = raw_dataset.map(_parse_fn)
  return dataset
