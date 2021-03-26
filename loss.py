class LabelSmoothingLoss():
    def __init__(self, smoothing=0.0):
        self.smoothing = smoothing

    def __call__(self, input, target):
        log_prob = input.log_softmax(-1)
        weight = input.new_ones(input.size()) * (self.smoothing / (input.size(-1) - 1.))
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(-1)
        return loss