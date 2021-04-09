import torch
from torch import nn


from model_architecture.cnn_embedder import CNNEmbedder
from model_architecture.pytorch_transformer.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from model_architecture.causal_transformer_decoder import (
    CausalTransformerDecoder,
    CausalTransformerDecoderLayer,
)


class Model(nn.Module):
    def __init__(self, bpe_num, N, n, ff, first_k, first_s, last_s,
                 enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers,
                 dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers,
                 dropout_ff = 0.1, dropout_dec_emb = 0.1,
                 max_trn_len=128, tta=None):
        super().__init__()
        self.dec_d_model = dec_d_model
        self.dec_num_layers= dec_num_layers
        self.tta = tta
        # CNN
        self.enc_emb = CNNEmbedder(enc_d_model,N,n,ff, first_k, first_s, last_s)
        # ADDITIONAL EMB
        self.add_emb = nn.Embedding(32,enc_d_model)
        # ENC
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=enc_d_model, nhead=enc_nhead, dim_feedforward=enc_dim_feedforward,
                                    dropout=dropout_ff),
            num_layers=enc_num_layers,
        )
        # ENC2DEC
        self.enc2dec = nn.Linear(enc_d_model, dec_d_model)
        # DEC
        self.dec_emb = nn.Embedding(bpe_num, dec_d_model)
        self.dec_emb_dropout = nn.Dropout(dropout_dec_emb)
        self.decoder = CausalTransformerDecoder(
            CausalTransformerDecoderLayer(d_model=dec_d_model, nhead=dec_nhead, dim_feedforward=dec_dim_feedforward,
                                          dropout=dropout_ff),
            num_layers=dec_num_layers,
            max_trn_len=max_trn_len,
        )
        self.dec_classifier = nn.Linear(dec_d_model, bpe_num)

        self.add_classifier = nn.Sequential(nn.Linear(enc_d_model, 512), nn.ReLU(),
                                            nn.Linear(512, 256), nn.ReLU(),
                                            nn.Linear(256, 192))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data)

    def forward(self, imgs_tensor, tgt_ids, dropout_p=0., dropout_h=0.):
        bs = len(imgs_tensor)
        # TTA - for validation
        with torch.cuda.amp.autocast(enabled=False):
            imgs_tensor = imgs_tensor if self.tta is None else self.tta(imgs_tensor)
        # cnn
        src = self.enc_emb(imgs_tensor)        # b, d_model, sqrt(i), sqrt(i)
        src = src.flatten(2,3).permute(2,0,1)  # i, b, d_model

        # add
        add = self.add_emb.weight.unsqueeze(1).repeat(1,bs,1)
        src = torch.cat([add, src], dim=0)

        # enc
        src = self.encoder(src, dropout_p=dropout_p, dropout_h=dropout_h)  # i, b, d_model

        # add
        add_out = self.add_classifier(src[:27]).transpose(0,1)

        # enc2dec
        src = self.enc2dec(src)

        # dec
        tgt = self.dec_emb(tgt_ids).permute(1, 0, 2)  # j, b, d_model
        tgt = self.dec_emb_dropout(tgt)
        dec = self.decoder(tgt,src, dropout_p=dropout_p, dropout_h=dropout_h)  # j, b, d_model
        dec_out = self.dec_classifier(dec)     # j, b, bpe_num
        dec_out = dec_out.transpose(0,1)       # b, j, bpe_num

        return dec_out, add_out


    def predict(self, imgs_tensor, max_pred_len=160):
        device = imgs_tensor.device
        bs = len(imgs_tensor)

        with torch.cuda.amp.autocast(enabled=False):
            imgs_tensor = imgs_tensor if self.tta is None else self.tta(imgs_tensor)

        src = self.enc_emb(imgs_tensor)
        src = src.flatten(2,3).permute(2,0,1)

        # enc
        src = self.encoder(src)
        # enc2dec
        src = self.enc2dec(src)

        decoded_ids = torch.ones(bs, 1, device=device).long()
        lens = torch.ones(bs, 1, device=device).long() * max_pred_len

        generated_ids = []
        cache = None
        # generation loop
        i = 0
        while ((lens < max_pred_len).sum() < bs) and i < max_pred_len:
            i += 1

            tgt = self.dec_emb(decoded_ids)
            tgt = tgt.permute(1, 0, 2)

            dec, cache = self.decoder(tgt, src, cache, causal=True)

            dec_out = self.dec_classifier(dec[-1, :, :])
            new_id = dec_out.argmax(1)
            generated_ids.append(new_id)
            decoded_ids = torch.cat([decoded_ids,new_id.unsqueeze(1).long()], dim=1,)

            lens[(new_id == 2)*(lens[:,0]==max_pred_len)] = i

        return generated_ids, lens

    def encoder_output(self, imgs_tensor):
        imgs_tensor = imgs_tensor if self.tta is None else self.tta(imgs_tensor)
        src = self.enc_emb(imgs_tensor)
        src = src.flatten(2,3).permute(2,0,1)
        src = self.encoder(src)
        src = self.enc2dec(src)
        return src

    def decoder_output(self, encoded, cache, decoded_tokens):
        decoded_embedding = self.dec_emb(decoded_tokens)
        decoded_embedding = decoded_embedding.permute(1, 0, 2)

        decoded, cache = self.decoder(decoded_embedding, encoded, cache, causal=True)

        logits = self.dec_classifier(decoded[-1, :, :])

        return logits.log_softmax(-1), cache
