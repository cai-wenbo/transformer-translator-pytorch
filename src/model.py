import torch
import torch.nn as nn
import math


"""
A large portion of the code is from: 
@misc{Gordić2020PyTorchOriginalTransformer,
  author = {Gordić, Aleksa},
  title = {pytorch-original-transformer},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/gordicaleksa/pytorch-original-transformer},
}
"""


class Transformer(nn.Module):
    def __init__(self, model_dimension, src_vocab_size, trg_vocab_size, number_of_heads, number_of_layers, dropout_probability = 0.1):
        super().__init__()

        #  embedding layer
        self.src_embedding = Embedding(src_vocab_size, model_dimension)
        self.trg_embedding = Embedding(trg_vocab_size, model_dimension)

        #  positional encoding
        self.src_pos_encoding = PositionalEncoding(model_dimension=model_dimension, dropout_probability=dropout_probability)
        self.trg_pos_encoding = PositionalEncoding(model_dimension=model_dimension, dropout_probability=dropout_probability)

        self.transformer_body = nn.Transformer(
                d_model            = model_dimension,
                nhead              = number_of_heads,
                num_encoder_layers = number_of_layers,
                num_decoder_layers = number_of_layers,
                dropout            = dropout_probability,
                )


        self.linear = nn.Linear(model_dimension, trg_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim = -1)

    def forward(self, b_text_src, b_text_trg, b_mask_src, b_mask_trg):
        src_embedding_batch = self.src_embedding(b_text_src)
        src_embedding_batch = self.src_pos_encoding(src_embedding_batch)

        trg_embedding_batch = self.trg_embedding(b_text_trg)
        trg_embedding_batch = self.trg_pos_encoding(trg_embedding_batch)

        transformer_out = self.transformer_body(
                src = src_embedding_batch,
                tgt = trg_embedding_batch,
                src_key_padding_mask = b_mask_src,
                tgt_key_padding_mask = b_mask_trg,
                memory_key_padding_mask = b_mask_src,
                tgt_is_causal = True
                )

        linear_out = self.linear(transformer_out)
        trg_log_probs = self.log_softmax(linear_out)
        return trg_log_probs





class Embedding(nn.Module):
    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dimension)
        self.scalar = math.sqrt(model_dimension)

    def forward(self, b_tokens):
        return self.embedding(b_tokens) * self.scalar



class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length = 5000):
        super().__init__()

        # (stated in the paper) Use sine functions whose frequencies form a geometric progression as position encodings,
        # (learning encodings will also work so feel free to change it!). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        # Register buffer because we want to save the positional encodings table inside state_dict even though
        # these are not trainable (not model's parameters) so they otherwise would be excluded from the state_dict
        self.register_buffer('positional_encodings_table', positional_encodings_table)
        self.dropout = nn.Dropout(dropout_probability)


    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # embedding_batch's shape = (B, S/T, D), where S/T max src/trg token-sequence length, D - model dimension
        # So here we get (S/T, D) shape which will get broad-casted to (B, S/T, D) when we try and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + positional_encodings)
