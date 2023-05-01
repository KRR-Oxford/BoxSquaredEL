import os.path

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
            'elem': ElbeLoadedModel,
            'EmELpp': ElbeLoadedModel,
            'boxel': BoxELLoadedModel
        }
        return model_dict[name].load(folder, embedding_size, device, best)


class BoxSqELLoadedModel(LoadedModel):
    class_embeds: torch.Tensor
    individual_embeds: torch.Tensor
    bumps: torch.Tensor
    individual_bumps: torch.Tensor
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
        if os.path.exists(f'{folder}/individual_embeds{suffix}.npy'):
            model.individual_embeds = torch.from_numpy(np.load(f'{folder}/individual_embeds{suffix}.npy')).to(device)
            model.individual_bumps = torch.from_numpy(np.load(f'{folder}/individual_bumps{suffix}.npy')).to(device)
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


class BoxELLoadedModel(LoadedModel):
    min_embedding: torch.Tensor
    delta_embedding: torch.Tensor
    relation_embedding: torch.Tensor
    scaling_embedding: torch.Tensor

    def is_translational(self):
        return False

    @staticmethod
    def load(folder, embedding_size, device, best=False):
        model = BoxELLoadedModel()
        model.embedding_size = embedding_size
        suffix = '_best' if best else ''
        model.min_embedding = torch.from_numpy(np.load(f'{folder}/min_embeds{suffix}.npy')).to(device)
        model.delta_embedding = torch.from_numpy(np.load(f'{folder}/delta_embeds{suffix}.npy')).to(device)
        model.relation_embedding = torch.from_numpy(np.load(f'{folder}/rel_embeds{suffix}.npy')).to(device)
        model.scaling_embedding = torch.from_numpy(np.load(f'{folder}/scaling_embeds{suffix}.npy')).to(device)
        return model
