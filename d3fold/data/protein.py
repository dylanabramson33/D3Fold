from itertools import repeat
import numpy as np
import random
import torch
from typing import Dict, Any

from omegaconf import DictConfig
import hydra
import pickle
import os

from d3fold.data.openfold.raw_protein import make_pdb_features
from d3fold.data.openfold.raw_protein import np_to_tensor_dict
from d3fold.data.openfold import transforms
from d3fold.data.openfold.raw_protein import RawProtein
from concurrent.futures import ThreadPoolExecutor


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
  
  
class ProteinData:
    def __init__(self, data: torch.Tensor, type_: ProteinDataType):
        self.data = data
        self.type_ = type_
        self.masked_data = None

    def get_filter_mask(self, filter_fn):
        if self.type_.pair_type:
            return filter_fn(self.data).any(dim=0)
        else:
            return filter_fn(self.data)

    def mask_data(self, mask):
        if self.type_.meta_data or self.type_.mask_template is None:
            return

        if isinstance(self.data, np.ndarray):
            mask = mask.numpy()
        self.data[mask] = self.type_.mask_template

    def crop_data(self, mask):
        if self.type_.meta_data:
            return

        if isinstance(self.data, np.ndarray):
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

class TorchProtein:
    def __init__(self):
        self._features = {}

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, ProteinData):
            self._features[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if name in self._features:
            return self._features[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any], type_dict: Dict[str, ProteinDataType]):
        protein = cls()
        for key, value in data_dict.items():
            if key in type_dict:
                setattr(protein, key, ProteinData(value, type_dict[key]))
        return protein

    def mask_fields(self, ignore_mask_fields=(), mask_percent=0.15):
        num_samples = len(self.aatype.data) * mask_percent
        num_samples = int(round(num_samples))
        indices = torch.arange(len(self.aatype.data))
        perm = torch.randperm(indices.size(0))
        mask = perm[:num_samples]
        for field, data in self._features.items():
            if field not in ignore_mask_fields:
                data.mask_data(mask)
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
        elif crop_strategy == "mix":
            crop_strategy = random.choice(list(crop_fns.keys()))
        
        mask = crop_fns[crop_strategy](crop_len)

        for field, data in self._features.items():
            if field not in ignore_mask_fields:
                data.crop_data(mask)

    def filter_fields(self, filter_fn, fields):
        masks = [self._features[field].get_filter_mask(filter_fn) for field in fields]
        mask = torch.stack(masks).any(dim=0)
        for data in self._features.values():
            data.crop_data(mask)

    def __repr__(self):
        return str({key: value for key, value in self._features.items()})
    
    @staticmethod
    def transform_features(feats):
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
        return tensor_dic

    @classmethod
    def from_pdb(cls, pdb_file, type_dict, chain_ids=None, save_path=None):
        try:
            protein = RawProtein.from_pdb_path(pdb_file, chain_ids)
            feats = make_pdb_features(protein, "no desc", is_distillation=False)
            tensor_dic = TorchProtein.transform_features(feats)
            torch_protein = cls.from_dict(tensor_dic, type_dict)
            
            if save_path:
                with open(save_path, "wb") as f:
                    pickle.dump(torch_protein, f)

            return torch_protein
        except Exception as e:
            print(f"Failed to load {pdb_file}")
            print(e)

    @classmethod
    def load_pdb_ids(cls, pdb_ids, all_chain_ids, type_dict, save_path=None, max_workers=5):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if save_path:
                save_paths = [os.path.join(save_path, f"{pdb_id}.pkl") for pdb_id in pdb_ids]
            
            list(executor.map(
                cls.from_pdb, pdb_ids, repeat(type_dict, len(pdb_ids)), all_chain_ids, save_paths))
