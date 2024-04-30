from dataclasses import dataclass

import random
import numpy as np
import torch
from biotite import structure
import biotite.structure.io.pdb as pdb
from data.residue_constants import THREE_TO_IND, THREE_TO_ONE



coord_mask = torch.ones(3) * torch.nan
COORD = ProteinDataType("COORD", pad_type="torch_geometric", mask_template=coord_mask)

seq_mask = torch.ones(1) * torch.nan
SEQ = ProteinDataType("SEQ", pad_type="seq", mask_template=seq_mask)

FRAME_R = ProteinDataType("FRAME_R", pad_type="seq", mask_template=None)
FRAME_T = ProteinDataType("FRAME_T", pad_type="seq", mask_template=None)

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
    def from_pdb(cls, pdb_path, chain_id=None):
        struct = pdb.PDBFile.read(pdb_path).get_structure(model=1)
        if chain_id is not None:
            struct = struct[struct.chain_id == chain_id]
            
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

    def mask_data(self, mask_prob=0.1, ignore_mask_fields=None):
        if  ignore_mask_fields is None:
          ignore_mask_fields = []
        mask = torch.rand(self.coords.data.shape[0]) < mask_prob
        for field in self.__dataclass_fields__.keys():
            if field in ignore_mask_fields:
              continue
            field_data = getattr(self, field)
            if type(field_data.data) is torch.Tensor:
              field_data.mask_data(mask)
            if type(field_data.data) is np.ndarray:
              np_mask = mask.numpy()
              field_data.mask_data(np_mask)

        return mask
    
    def random_crop_mask(self, crop_len=400):
        start = random.randint(0, len(self.coords.data) - crop_len)
        end = start + crop_len
        mask = torch.zeros(len(self.coords.data))
        mask[start:end] = 1
        return mask.bool()

    def center_crop_mask(self, crop_len=400):
        start = len(self.coords.data) // 2 - crop_len // 2
        end = start + crop_len
        mask = torch.zeros(len(self.coords.data))
        mask[start:end] = 1
        return mask.bool()

    def crop_data(self, crop_strategy="mix", crop_len=400):
      if len(self.coords.data) < crop_len:
        return
      
      crop_fns = {
        "random": self.random_crop_mask,
        "center": self.center_crop_mask
      }

      if crop_strategy == "random":
        mask = crop_fns[crop_strategy](crop_len)
      elif crop_strategy == "center":
        mask = crop_fns[crop_strategy](crop_len)
      elif crop_strategy == "mix":
        # randomly select strategy
        crop_strategy = random.choice(list(crop_fns.keys()))
        mask = crop_fns[crop_strategy](crop_len)
      else:
        raise ValueError("Invalid crop strategy")
      
      for field in self.__dataclass_fields__.keys():
        field_data = getattr(self, field)
        if type(field_data.data) is torch.Tensor:
          field_data.data = field_data.data[mask]
        if type(field_data.data) is np.ndarray:
          np_mask = mask.numpy()
          field_data.data = field_data.data[np_mask]
