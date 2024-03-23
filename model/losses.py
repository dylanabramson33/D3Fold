import torch.nn as nn
class LossFn:
    def __init__(self, loss_fn, representation_target="pair"):
        self.loss_fn = loss_fn
        self.representation_target = representation_target

    def __call__(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

class PairwiseLoss(LossFn):
    def __init__(self, loss_fn, representation_target="pair"):
        super().__init__(loss_fn, representation_target=representation_target)

    def __call__(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

class SequenceLoss(LossFn):
    def __init__(self, loss_fn, representation_target="seq"):
        super().__init__(loss_fn, representation_target=representation_target)

    def __call__(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

class StructuredLoss(LossFn):
    def __init__(self, loss_fn, representation_target="structure"):
        super().__init__(loss_fn, representation_target=representation_target)

    def __call__(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

def masked_sequence_loss(y_pred, y_true, mask):
    return nn.CrossEntropyLoss()(y_pred, y_true, mask)

contact_loss = PairwiseLoss(nn.BCEWithLogitsLoss())
sequence_loss = SequenceLoss(masked_sequence_loss)
