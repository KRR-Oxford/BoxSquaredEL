import torch
import numpy as np
from boxes import Boxes
from abc import ABC, abstractmethod


class LoadedModel(ABC):
    embedding_size: int

    def get_boxes(self, embedding):
        return Boxes(embedding[:, :self.embedding_size], torch.abs(embedding[:, self.embedding_size:]))

    @abstractmethod
    def is_translational(self):
        pass

    @staticmethod
    def from_name(name, folder, embedding_size, device, best=False):
        model_dict = {
            'boxsqel': BoxSqELLoadedModel,
            'elbe': ElbeLoadedModel,
            'elbe+': ElbeLoadedModel,
            'elem': ElbeLoadedModel
        }
        return model_dict[name].load(folder, embedding_size, device, best)


class BoxSqELLoadedModel(LoadedModel):
    class_embeds: torch.Tensor
    bumps: torch.Tensor
    relation_heads: torch.Tensor
    relation_tails: torch.Tensor

    def is_translational(self):
        return False

    @staticmethod
    def load(folder, embedding_size, device, best=False):
        model = BoxSqELLoadedModel()
        model.embedding_size = embedding_size
        suffix = '_best' if best else ''
        model.class_embeds = torch.from_numpy(np.load(f'{folder}/class_embeds{suffix}.npy')).to(device)
        model.bumps = torch.from_numpy(np.load(f'{folder}/bumps{suffix}.npy')).to(device)
        model.relation_heads = torch.from_numpy(np.load(f'{folder}/rel_heads{suffix}.npy')).to(device)
        model.relation_tails = torch.from_numpy(np.load(f'{folder}/rel_tails{suffix}.npy')).to(device)
        return model


class ElbeLoadedModel(LoadedModel):
    class_embeds: torch.Tensor
    relation_embeds: torch.Tensor

    def is_translational(self):
        return True

    @staticmethod
    def load(folder, embedding_size, device, best=False):
        model = ElbeLoadedModel()
        model.embedding_size = embedding_size
        suffix = '_best' if best else ''
        model.class_embeds = torch.from_numpy(np.load(f'{folder}/class_embeds{suffix}.npy')).to(device)
        model.relation_embeds = torch.from_numpy(np.load(f'{folder}/relations{suffix}.npy')).to(device)
        return model
