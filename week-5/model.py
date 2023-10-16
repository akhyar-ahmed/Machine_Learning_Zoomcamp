import pickle
import os

class Model:
  def __init__(self, model_path, dv_path):
    self.model = self.load_model(model_path)
    self.dv = self.load_dv(dv_path)
  
  def load_model(self, model_path):
    model = ""
    with open(model_path, "rb") as f_in:
      model = pickle.load(f_in)
    return model

  def load_dv(self, dv_path):
    dv = ""
    with open(dv_path, "rb") as f_in:
      dv = pickle.load(f_in)
    return dv