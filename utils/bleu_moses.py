# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BLEU metric implementation.
"""

import os
import re
import subprocess
import numpy as np
import tensorflow as tf


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.

    Args:
      hypotheses: A numpy array of strings where each string is a single example.
      references: A numpy array of strings where each string is a single example.
      lowercase: If true, pass the "-lc" flag to the multi-bleu script

    Returns:
      The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Alternatively, get file locally
    training_dir = os.path.dirname(os.path.realpath(__file__))
    multi_bleu_path = os.path.join(training_dir, "multi-bleu.perl")

    hypothesis_name = os.path.join(training_dir, "hypothesis.txt")
    reference_name = os.path.join(training_dir, "reference.txt")
    with open(hypothesis_name, 'w', encoding='utf8') as hypothesis_file:
        hypothesis_file.write("\n".join(hypotheses))
    with open(reference_name, 'w', encoding='utf8') as reference_file:
        reference_file.write("\n".join(references))

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_name, "r", encoding='utf8') as read_pred:
        # bleu_cmd = ["c:/perl64/bin/perl"]
        bleu_cmd = ["perl"]
        bleu_cmd += [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_name]
        try:
            bleu_out = subprocess.check_output(
                bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                tf.logging.warning("multi-bleu.perl script returned non-zero exit code")
                tf.logging.warning(error.output)
            bleu_score = np.float32(0.0)

    return np.float32(bleu_score)
