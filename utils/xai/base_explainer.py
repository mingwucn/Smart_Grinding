import torch
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    def __init__(self, model):
        self.model = model
    
    @abstractmethod
    def explain(self, input_dict, target_output_idx=None):
        pass
    
    @abstractmethod
    def visualize(self, explanation, input_data):
        pass
    
    @abstractmethod
    def metrics(self, explanation):
        pass
