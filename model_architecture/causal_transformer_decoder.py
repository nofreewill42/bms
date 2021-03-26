import torch
import torch.nn as nn


class CausalTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, max_trn_len):
        super().__init__(decoder_layer, num_layers)
        future_mask = generate_future_mask(max_trn_len)
        self.register_buffer('future_mask', future_mask)

    def forward(self, tgt, src, cache=None, causal=False):

        output = tgt

        # Loss
        if not causal:
            for mod in self.layers:
                output = mod(output, src, self.future_mask[:tgt.size(0),:tgt.size(0)])
            return output

        # Generate
        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, src, causal=True)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache

class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt, src, tgt_mask=None, causal=False):

        if not causal:
            return super().forward(tgt, src, tgt_mask=tgt_mask)

        tgt_last_tok = tgt[-1:, :, :]

        # self attention
        tmp_tgt = self.self_attn(tgt_last_tok, tgt, tgt)[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # encoder-decoder attention
        if src is not None:
            tmp_tgt = self.multihead_attn(tgt_last_tok, src, src)[0]
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        # feed-forward
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok


def generate_future_mask(max_len):
    mask = torch.tril(torch.ones(max_len, max_len))*-1+1
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask
