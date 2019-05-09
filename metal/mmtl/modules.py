import torch.nn as nn


class MetalModule(nn.Module):
    """An abstract class of a module that accepts and returns a dict"""

    def __init__(self):
        super().__init__()


class MetalModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, X):
        # The object that is passed out must be different from the object that gets
        # passed in so that cached outputs from intermediate modules aren't mutated
        if is_input:
            half_width_of_sequence = X["data"].shape[1] / 2
            if half_width_of_sequence % 2 != 0:
                raise "Token input of length n must be of form n input ids and n segment ids"
            half_width_of_sequence = int(half_width_of_sequence)
            X_out = {k: v for k, v in X.items()}

            input_ids = X["data"][:, :half_width_of_sequence]
            token_type_ids = X["data"][:, half_width_of_sequence:]
            X_out["data"] = self.module(input_ids=input_ids, token_type_ids=token_type_ids)
            return X_out
        else:
            X_out = {k: v for k, v in X.items()}
            X_out["data"] = self.module(X["data"])
            return X_out
