from dataclasses import dataclass
import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from d3fold.data.openfold.raw_protein import make_pdb_features
from d3fold.data.openfold.raw_protein import np_to_tensor_dict
from d3fold.data.openfold import transforms
from d3fold.data.openfold.raw_protein import RawProtein

class ProteinDataType:
    def __init__(
          self,
          type=None,
          pad_type="torch_geometric",
          mask_template=None,
          meta_data=False,
          pair_type=False,
          ):
        self.type = type
        self.mask_template = mask_template
        self.pad_type = pad_type
        self.meta_data = meta_data
        self.pair_type = pair_type


    def __repr__(self):
        return self.type

    def __str__(self):
        return self.type

@dataclass
class ProteinData():
    data: torch.Tensor
    type_: ProteinDataType
    masked_data: torch.Tensor = None

    def get_filter_mask(self, filter_fn):
      if self.type_.pair_type:
        return filter_fn(self.data).any(dim=0)
      else:
        return filter_fn(self.data)

    def mask_data(self, mask):
        if self.type_.meta_data or self.type_.mask_template is None:
          return

        if type(self.data) is np.ndarray:
          mask = mask.numpy()
        self.data[mask] = self.type_.mask_template

    def crop_data(self, mask):
      if self.type_.meta_data:
        return

      if type(self.data) is np.ndarray:
        mask = mask.numpy()

      if self.type_.pair_type:
         self.data = self.data[mask]
         self.data = self.data[:, mask]
      else:
        self.data = self.data[mask]

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)


mask_templates = {
   "sequence" : np.array(["<mask>"])
}

@hydra.main(version_base=None, config_path=".", config_name="config")
def build_types(cfg: DictConfig):
  types = {}
  features = cfg.features
  for feature in features:
    apply_mask = feature.get("apply_mask", False)
    type_ = ProteinDataType(
      type=feature.name,
      pad_type=feature.pad_type,
      meta_data=feature.get("meta_data", False),
      pair_type=feature.get("pair_type", False),
      mask_template=mask_templates[feature.name] if apply_mask else None
    )
    types[feature.name] = type_
  
  return types
  

@dataclass
class TorchProtein:
    def __init__(self, **kwargs):
      for key in kwargs.keys():
        setattr(self, key, kwargs[key])

    @classmethod
    def from_dict(cls, data_dict, type_dict):
      data_dict = {key: ProteinData(data_dict[key], type_dict[key]) for key in type_dict.keys()}
      return cls(**data_dict)

    def mask_fields(self, ignore_mask_fields=(), mask_percent=0.15):
      num_samples = len(self.aatype.data) * mask_percent
      num_samples = int(round(num_samples))
      indices = torch.arange(len(self.aatype.data))
      perm = torch.randperm(indices.size(0))
      mask = perm[:num_samples]
      for field in self.__dataclass_fields__.keys():
        if field in ignore_mask_fields:
            continue
        field_data = getattr(self, field)
        if not isinstance(field_data, type(None)):

          field_data.mask_data(mask)

      return mask

    def random_crop_mask(self, crop_len=400):
      start = random.randint(0, len(self.aatype.data) - crop_len)
      end = start + crop_len
      mask = torch.zeros(len(self.aatype.data))
      mask[start:end] = 1
      return mask.bool()

    def center_crop_mask(self, crop_len=400):
        start = len(self.aatype.data) // 2 - crop_len // 2
        end = start + crop_len
        mask = torch.zeros(len(self.aatype.data))
        mask[start:end] = 1
        return mask.bool()

    def crop_fields(self, ignore_mask_fields=(), crop_strategy="mix", crop_len=400):
      crop_fns = {
        "random": self.random_crop_mask,
        "center": self.center_crop_mask
      }
      if len(self.aatype.data) < crop_len:
        return
      elif crop_strategy == "random":
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
        if field in ignore_mask_fields or isinstance(getattr(self, field), type(None)):
            continue
        field_data = getattr(self, field)
        field_data.crop_data(mask)

    def filter_fields(self, filter_fn, fields):
      masks = []
      for field in fields:
        field_data = getattr(self, field)
        mask = field_data.get_filter_mask(filter_fn)
        masks.append(mask)
      # take intersection of all masks
      mask = torch.stack(masks).any(dim=0)
      for field in self.__dataclass_fields__.keys():
        if isinstance(getattr(self, field), type(None)):
          continue
        field_data = getattr(self, field)
        field_data.crop_data(mask)

    @classmethod
    def from_pdb(cls, pdb_file, type_dict):
      protein = RawProtein.from_pdb_path(pdb_file)
      feats = make_pdb_features(protein, "no desc", is_distillation=False)
      tensor_dic = np_to_tensor_dict(feats, feats.keys())
      tensor_dic = transforms.squeeze_features(tensor_dic)
      tensor_dic = transforms.make_atom14_masks(tensor_dic)
      tensor_dic = transforms.make_atom14_positions(tensor_dic)
      tensor_dic = transforms.atom37_to_frames(tensor_dic)
      tensor_dic = transforms.atom37_to_torsion_angles(tensor_dic)
      tensor_dic = transforms.make_pseudo_beta(tensor_dic)
      tensor_dic = transforms.get_backbone_frames(tensor_dic)
      tensor_dic = transforms.get_chi_angles(tensor_dic)
      tensor_dic = transforms.get_distance_mat_stack(tensor_dic)
      tensor_dic = transforms.convert_angles_to_degrees(tensor_dic)
      tensor_dic = transforms.get_quantized_phi_psi_omega(tensor_dic)
      tensor_dic = transforms.relative_positions(tensor_dic)
      return cls.from_dict(tensor_dic, type_dict)