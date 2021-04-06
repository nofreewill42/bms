import torch


class BeamSearcher:
    def __init__(self, models, weights, tfmss, bpe_num, beam_width=4, max_len=160):
        self.models = models
        self.weights = weights
        self.tfmss = tfmss
        self.bw = beam_width
        self.max_len = max_len
        self.bpe_num = bpe_num

    def eval(self):
        for model in self.models:
            model.eval()
    def train(self):
        for model in self.models:
            model.train()

    def predict(self, imgs_tensor, max_pred_len=256):
        bs = len(imgs_tensor)
        device = imgs_tensor.device
        bw = self.bw
        max_len = max_pred_len
        bpe_num = self.bpe_num
        eos_id = 2

        # Initialize

        enc_outs = [model.encoder_output(self.tfmss[i](imgs_tensor)) for i, model in enumerate(self.models)]
        #enc_outs = [model.encoder_output(imgs_tensor) for i, model in enumerate(self.models)]

        start_tokens = torch.ones(bs, 1, device=device, dtype=torch.long)
        caches = [None]*len(self.models)
        route_probs = torch.zeros(bs * bw, 1, device=device)

        beam_ids   = torch.zeros(bs, max_len, device=device, dtype=torch.long)
        beam_probs = torch.full((bs,), float('-inf'), device=device)
        beam_lens  = torch.ones(bs, device=device, dtype=torch.long)
        done_beams = torch.zeros(bs, device=device, dtype=torch.bool)

        # First decoder output

        bpe_probses, caches = list(zip(*[self.models[i].decoder_output(enc_outs[i], caches[i], start_tokens)
                                         for i in range(len(self.models))]))
        bpe_probs = sum([self.weights[i] * bpe_probses[i] for i in range(len(self.models))])
        topk_probs, topk_idxs = bpe_probs.topk(bw, dim=1)
        route_probs = topk_probs.reshape(bs * bw, 1)

        enc_outs = [enc_outs[i].unsqueeze(2).repeat(1,1,bw,1).reshape(-1,bs*bw,self.models[i].dec_d_model)
                    for i in range(len(self.models))]
        caches = [caches[i].unsqueeze(3).repeat(1,1,1,bw,1)\
                      .reshape(self.models[i].dec_num_layers,-1,bs*bw,self.models[i].dec_d_model)
                  for i in range(len(self.models))]
        decoded_tokens = torch.cat([torch.ones(bs * bw, 1).long().to('cuda:0'), topk_idxs.reshape(bs * bw, 1)], dim=1)

        # Loop
        while True:
            route_len = decoded_tokens.shape[1]
            if route_len == max_len:
                break

            bpe_probses, caches = list(zip(*[self.models[i].decoder_output(enc_outs[i], caches[i], decoded_tokens)
                                             for i in range(len(self.models))]))
            bpe_probs = sum([self.weights[i] * bpe_probses[i] for i in range(len(self.models))])
            route_bpe_probs = route_probs + bpe_probs

            # check dones
            best_bpe_probs = route_bpe_probs.reshape(bs, bw * bpe_num).max(dim=1)[0]
            done_beams = beam_probs > best_bpe_probs

            if done_beams.all():
                break

            # step
            topk_probs, topk_idxs = route_bpe_probs.reshape(bs, bw * bpe_num).topk(bw, dim=1)

            index_helper = torch.arange(bs, device=device)
            shift_idxs = (index_helper * bw).reshape(bs, 1).repeat(1, bw).reshape(bs * bw)
            is_eos, eos_idxs = ((topk_idxs%bpe_num)==eos_id).max(dim=1)
            is_better = (topk_probs[index_helper,eos_idxs] > beam_probs) * is_eos
            beam_probs[is_better] = topk_probs[is_better,eos_idxs[is_better]]
            beam_lens[is_better] = route_len
            beam_ids[is_better,:route_len] = decoded_tokens[index_helper*bw+topk_idxs[index_helper,eos_idxs]//bpe_num][is_better]

            # route_probs
            route_probs = topk_probs.reshape(bs * bw, 1)
            keep_idxs = shift_idxs + (topk_idxs // bpe_num).reshape(bs * bw)
            caches = [cache[:,:,keep_idxs] for cache in caches]
            decoded_tokens = decoded_tokens[keep_idxs]
            decoded_tokens = torch.cat([decoded_tokens, (topk_idxs % bpe_num).reshape(bs * bw, 1)], dim=1)

        return beam_ids, beam_lens

