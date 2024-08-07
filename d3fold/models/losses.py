import torch
import torch.nn as nn

from d3fold.data.openfold import residue_constants as rc

class LossFn:
    def __init__(self, loss_fn, representation_target="pair", name=None):
        self.loss_fn = loss_fn
        self.representation_target = representation_target
        self.name = name

    def __call__(self, y_pred, y_true, **kwargs):
        return self.loss_fn(y_pred, y_true, **kwargs)

class PairwiseLoss(LossFn):
    def __init__(self, loss_fn, representation_target="pair", name=None):
        super().__init__(loss_fn, representation_target=representation_target, name=name)

class SequenceLoss(LossFn):
    def __init__(self, loss_fn, representation_target="seq", name=None):
        super().__init__(loss_fn, representation_target=representation_target, name=name)


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

def sequence_loss(y_pred, data):
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    B, R = y_pred.shape[0], y_pred.shape[1]
    mask = torch.zeros(B, R, dtype=torch.bool, device=y_pred.device)
    mask.scatter_(1, data.mask, True)
    target = data.aatype
    target = target.masked_fill(~mask, -100)
    y_pred = y_pred.permute(0, 2, 1)
    return loss_fn(y_pred, target)

pairwise_loss = PairwiseLoss(pairwise_dist_loss, name="pairwise_dist_loss")
seq_loss = SequenceLoss(sequence_loss, name="sequence_mask_loss")