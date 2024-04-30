from dataclasses import dataclass

import random
import numpy as np
import torch

class ProteinDataType:
    def __init__(self, type=None, pad_type="torch_geometric", mask_template=None):
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
    masked_data: torch.Tensor = None

    def mask_data(self, mask):
        if self.type_.mask_template is None:
            return self.data
        if type(mask) is np.ndarray:
          self.masked_data = self.data.copy()
        elif type(mask) is torch.Tensor:
           self.masked_data = self.data.clone()
        self.masked_data = self.masked_data[mask]
        self.data[mask] = self.type_.mask_template

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)
    

coord_mask = torch.ones(3) * torch.nan
COORD = ProteinDataType("COORD", pad_type="torch_geometric", mask_template=coord_mask)

seq_mask = torch.ones(1) * torch.nan
SEQ = ProteinDataType("SEQ", pad_type="seq", mask_template=seq_mask)

FRAME_R = ProteinDataType("FRAME_R", pad_type="seq", mask_template=None)
FRAME_T = ProteinDataType("FRAME_T", pad_type="seq", mask_template=None)

raw_mask = np.array(["<mask>"])
RAW_SEQ = ProteinDataType("RAW_SEQ", pad_type="esm", mask_template=raw_mask)


@dataclass
class TorchProtein:
    coords: ProteinData
    seq: ProteinData
    frames_R: ProteinData
    frames_t: ProteinData
    raw_seq: ProteinData
    
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
