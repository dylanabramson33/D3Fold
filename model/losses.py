import torch
import torch.nn as nn

from D3Fold.data.openfold import residue_constants as rc


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

def pairwise_dist_loss(y_pred, data):
    loss_fn = nn.CrossEntropyLoss()
    distance_mat = data["distance_mat_stack"]
    gathered = distance_mat.argmax(dim=-1).long()
    CA_INDEX = rc.atom_types.index("CA")
    mask = data["atom37_atom_exists"][:, :, CA_INDEX]
    mask = torch.where(~mask.isnan(),mask,torch.zeros_like(mask)).bool()
    gathered[~mask.unsqueeze(-1).expand_as(gathered)] = -100
    y_pred = y_pred.permute(0,3,1,2)
    return loss_fn(y_pred, gathered)

pairwise_loss = PairwiseLoss(pairwise_dist_loss)
