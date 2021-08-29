import torch

class CachedLookup(object):

    def __init__(self, data):
        self.orig = data
        self.key = None
        self.current = None

    def __getitem__(self, key):
        assert not isinstance(key, slice), "CachedLookup does not support slicing, " \
                                           "you can slice the result of an index operation instead"

        assert torch.is_tensor(key)  # If tensor, idx all tensors by this tensor:

        if self.key is None:
            self.key = key
            self.current = self.orig[key]
        elif len(key) != len(self.key) or (key != self.key).any():
            self.key = key
            self.current = self.orig[key]

        return self.current
