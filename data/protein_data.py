from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from biotite import structure
import biotite.structure.io.pdb as pdb
from biotite.structure import AtomArray as AA

from data.constants import THREE_TO_IND, THREE_TO_ONE

def convert_to_resolution(struct, res="CA"):
    chain_data = struct[~struct.hetero]
    if res == "CA":
      chain_data = chain_data[chain_data.atom_name == "CA"]
    elif res == "backbone":
      chain_data = chain_data[structure.filter_backbone(chain_data)]

    return chain_data

def convert_to_tensor(struct, res="CA"):
    chain_data = convert_to_resolution(struct, res)
    coords = torch.tensor(chain_data.coord)
    seq = chain_data.res_name
    seq = torch.tensor([THREE_TO_IND[x] for x in seq])
    
    return coords, seq

def rigid_from_3_points(x1,x2,x3):
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / torch.norm(v1)
    u2 = v2 - e1 * (e1 @ v2)
    e2 = u2 / torch.norm(u2)
    e3 = torch.cross(e1, e2)
    R = torch.stack([e1, e2, e3], dim=1)
    t = x2
    return R, t

def get_ipa_mask(backbone_struct, ca_struct):
  res_ids = ca_struct.res_id

  clean_residues = []
  clean_backbone = []
  current_backbone_ind = 0
  for i,res_id in enumerate(res_ids):
    block = backbone_struct[backbone_struct.res_id == res_id]
    if block.atom_name[0] == "N" and block.atom_name[1] == "CA" and block.atom_name[2] == "C":
      clean_residues.append(i)
      clean_backbone.extend([current_backbone_ind,current_backbone_ind+1,current_backbone_ind+2])

    current_backbone_ind += len(block)
  backbone_mask = torch.zeros(len(backbone_struct.coord))
  residue_mask = torch.zeros(len(ca_struct.coord))
  backbone_mask[clean_backbone] = 1
  residue_mask[clean_residues] = 1

  return backbone_mask.bool(), residue_mask.bool()

def construct_ipa_frames(backbone_struct, ca_struct):
    backbone_mask, residue_mask = get_ipa_mask(backbone_struct, ca_struct)
    backbone_struct = torch.tensor(backbone_struct.coord)
    backbone_struct = backbone_struct[backbone_mask]
    frames = []
    for i in range(0,len(backbone_struct)-2,3):
        R, t = rigid_from_3_points(backbone_struct[i], backbone_struct[i+1], backbone_struct[i+2])
        frames.append((R, t))

    return frames, residue_mask

class ProteinDataType:
    def __init__(self, type=None, pad_type="torch_geo", mask_template=None):
        self.type = type
        self.mask_template = mask_template
        self.pad_type = pad_type

    def __repr__(self):
        return self.type

    def __str__(self):
        return self.type

@dataclass
class ProteinData():
    data: torch.Tensor
    type_: ProteinDataType

    def mask_data(self, mask):
        if self.type_.mask_template is None:
            return self.data

        self.data[mask] = self.type_.mask_template

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

coord_mask = torch.ones(3) * torch.nan
COORD = ProteinDataType("COORD", pad_type="torch_geometric", mask_template=coord_mask)

seq_mask = torch.ones(1) * torch.nan
SEQ = ProteinDataType("SEQ", pad_type="seq", mask_template=seq_mask)

FRAME_R = ProteinDataType("FRAME_R", pad_type="torch_geometric", mask_template=None)
FRAME_T = ProteinDataType("FRAME_T", pad_type="torch_geometric", mask_template=None)

raw_mask = np.array(["<mask>"])
RAW_SEQ = ProteinDataType("RAW_SEQ", pad_type="esm", mask_template=raw_mask)

@dataclass
class Chain:
    coords: ProteinData
    seq: ProteinData
    frames_R: ProteinData
    frames_t: ProteinData
    raw_seq: ProteinData

    @classmethod
    def from_pdb(cls, pdb_path):
        struct = pdb.PDBFile.read(pdb_path).get_structure(model=1)
        backbone_struct = convert_to_resolution(struct, res="backbone")
        ca_struct = convert_to_resolution(backbone_struct, res="CA")
        raw_seq = [THREE_TO_ONE[x.res_name] for x in ca_struct]
        raw_seq = np.array(raw_seq, dtype='object')
        coords, seq = convert_to_tensor(ca_struct)
        frames, residue_mask = construct_ipa_frames(backbone_struct, ca_struct)

        frames_R = torch.stack([f[0] for f in frames])
        frames_t = torch.stack([f[1] for f in frames])
        # remove broken indices
        coords = coords[residue_mask]
        seq = seq[residue_mask]
        frames_R = frames_R[residue_mask]
        frames_t = frames_t[residue_mask]

        coords = ProteinData(coords, COORD)
        seq = ProteinData(seq, SEQ)
        frames_R = ProteinData(frames_R, FRAME_R)
        frames_t = ProteinData(frames_t, FRAME_T)
        raw_seq = ProteinData(raw_seq, RAW_SEQ)
        return cls(coords, seq, frames_R, frames_t, raw_seq)

    def mask_data(self, mask_prob=0.1):
        mask = torch.rand(self.coords.data.shape[0]) < mask_prob
        for field in self.__dataclass_fields__.keys():
            field_data = getattr(self, field)
            if type(field_data.data) is torch.Tensor:
              field_data.mask_data(mask)
            if type(field_data.data) is np.ndarray:
              np_mask = mask.numpy()
              field_data.mask_data(np_mask)
        return self
            