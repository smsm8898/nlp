import torch

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(
            self,
            vocab_size:int, 
            pad_idx:int=0, 
            smoothing:float=0.1
        ):
        super().__init__()
        self.criterion = torch.nn.KLDivLoss(reduction="mean")
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0

            mask = (target == self.pad_idx).unsqueeze(1)
            true_dist.masked_fill_(mask, 0)

        return self.criterion(pred, true_dist)