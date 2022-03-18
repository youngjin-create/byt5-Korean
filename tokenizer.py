# coding=utf-8
#
# Everdoubling LLC.
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
#
# The following code is modified from HuggingFace's ByT5 Tokenizer: transformers/models/byt5/tokenization_byt5.py
#
""" Tokenization class for model ByT5."""


import warnings
from typing import Dict, List, Optional, Tuple, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.models.byt5.tokenization_byt5 import ByT5Tokenizer

class ByT5KoreanTokenizer(PreTrainedTokenizer):
    """
    Construct a ByT5Korean tokenizer.
    On top of ByT5's simple raw bytes utf-8 encoding, ByT5Korean adds extra tokens for Korean jamo.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        extra_ids (:obj:`int`, `optional`, defaults to 100):
            Add a number of extra ids added to the end of the vocabulary for use as sentinels. These tokens are
            accessible as "<extra_id_{%d}>" where "{%d}" is a number between 0 and extra_ids-1. Extra tokens are
            indexed from the end of the vocabulary up to beginning ("<extra_id_0>" is the last token in the vocabulary
            like in ByT5 preprocessing see `here
            <https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117>`__).
        additional_special_tokens (:obj:`List[str]`, `optional`):
            Additional special tokens used by the tokenizer.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=57,
        additional_special_tokens=None,
        **kwargs
    ) -> None:
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to ByT5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self._extra_ids = extra_ids

        # Add the special tokens (including extra_ids)
        for token in self.all_special_tokens:
            self.tokens_trie.add(token)

        self._utf_vocab_size = 2 ** 8  # utf is 8 bits
        self._utf_vocab_size += 19 + 21 + 28  # korean jamo

        # define special tokens dict
        self.special_tokens_encoder: Dict[int, str] = {
            self.pad_token: 0,
            self.eos_token: 1,
            self.unk_token: 2,
        }
        self._num_special_tokens = len(self.special_tokens_encoder)
        n = len(additional_special_tokens)
        for i, token in enumerate(additional_special_tokens):
            self.special_tokens_encoder[token] = self.vocab_size + i - n
        self.special_tokens_decoder: Dict[str, int] = {v: k for k, v in self.special_tokens_encoder.items()}

    @property
    def vocab_size(self):
        return self._utf_vocab_size + self._num_special_tokens + self._extra_ids

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. ByT5 does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: ``X </s>``
        - pair of sequences: ``A </s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def _convert_char_to_tokens_Korean(self, c):
        o = ord(c)
        if 44032 <= o and o <= 55203: # 44032: 가, 55203: 힣 
            o -= 44032
            return [chr(256 + (o // 588)), chr(256 + 19 + ((o % 588) // 28)), chr(256 + 19 + 21 + (o % 28))]
        return [chr(i) for i in c.encode("utf-8")]

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        if text in self.all_special_tokens:
            return [text]
            # return [self.special_tokens_encoder[text]]
        # tokens = [chr(i) for i in text.encode("utf-8")]
        # return tokens
        return sum([self._convert_char_to_tokens_Korean(c) for c in text], [])

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.special_tokens_encoder:
            token_id = self.special_tokens_encoder[token]
        elif token in self.added_tokens_encoder:
            token_id = self.added_tokens_encoder[token]
        # else:
            # token_id = token + self._num_special_tokens
        elif len(token) != 1:
            token_id = self.unk_token_id
        else:
            token_id = ord(token) + self._num_special_tokens
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.special_tokens_decoder:
            token = self.special_tokens_decoder[index]
        else:
            token = chr(index - self._num_special_tokens)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""
        ids = [ord(t[0]) for t in tokens]
        for i in range(len(ids)-2):
            if 256 <= ids[i] and ids[i] < 256+19 and 256+19 <= ids[i+1] and ids[i+1] < 256+19+21 and 256+19+21 <= ids[i+2] and ids[i+2] < 256+19+21+28:
                tokens[i] = chr(44032 + (ids[i]-256)*21*28 + (ids[i+1]-256-19)*28 + (ids[i+2]-256-19-21))
                tokens[i+1] = None
                tokens[i+2] = None
        for token in tokens:
            if token == None:
                continue
            if token in self.special_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8")
            elif token in self.added_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8")
            elif token in self.special_tokens_encoder:
                tok_string = token.encode("utf-8")
            elif token in self.added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                if type(token) == str and ord(token) >= 256:
                    tok_string = token.encode("utf-8")
                else:
                    tok_string = bytes([ord(token) if type(token) == str else min(255, token)])
            bstring += tok_string
        string = bstring.decode("utf-8", errors="ignore")
        return string

    # ByT5KoreanTokenizer has no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()


if __name__ == "__main__":
    tokenizer = ByT5KoreanTokenizer()
    text = "This is a test <extra_id_0> of the 가나힣 안녕하세요 <extra_id_1>."
    tokenized_text = tokenizer.tokenize(text)
    print(tokenized_text)
    print(tokenizer(text))
    print(tokenizer.convert_tokens_to_ids(tokenized_text))
    print(tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(tokenized_text)))
    print(tokenizer.convert_tokens_to_string(tokenized_text))