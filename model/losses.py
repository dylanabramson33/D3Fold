import torch.nn as nn

class LossFn:
    def __init__(self, loss_fn, representation_target="pair"):
        self.loss_fn = loss_fn
        self.representation_target = representation_target

class PairwiseLoss(LossFn):
    def __init__(self, loss_fn, representation_target="pair"):
        super().__init__(loss_fn, representation_target=representation_target)

    def __call__(self, y_pred, y_true, **kwargs):
        return self.loss_fn(y_pred, y_true, **kwargs)

class SequenceLoss(LossFn):
    def __init__(self, loss_fn, representation_target="seq"):
        super().__init__(loss_fn, representation_target=representation_target)

    def __call__(self, y_pred, y_true, **kwargs):
        return self.loss_fn(y_pred, y_true, **kwargs)

def masked_sequence_loss(y_pred, y_true, mask=None):
    loss_fn = nn.CrossEntropyLoss()
    y_true[~mask] = -100
    y_true = y_true.long()
    y_pred = y_pred.permute(0, 2, 1)

    return loss_fn(y_pred, y_true)

sequence_loss = SequenceLoss(masked_sequence_loss)
