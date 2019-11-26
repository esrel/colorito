from colorito.exceptions import SaveError, LoadError

import torch.nn as nn
import torch
import pickle
import os


class SmartModule(nn.Module):

    def __init__(self, *args):
        super(SmartModule, self).__init__()
        self.init_params = args

    @classmethod
    def name(cls):
        return cls.__name__

    @classmethod
    def metadata_fname(cls):
        return f'{cls.name()}.mdata.pl'

    @classmethod
    def parameters_fname(cls):
        return f'{cls.name()}.param.pt'

    def save(self, to):
        if not os.path.isdir(to):
            raise SaveError(
                cls=self.name(),
                err=f'{to} is not '
                    f'a directory.'
            )

        metadata_f = os.path.join(
         to, self.metadata_fname()
        )
        parameters_f = os.path.join(
         to, self.parameters_fname()
        )
        # save metadata:
        try:
            pickle.dump(
                self.init_params,
                open(metadata_f, 'wb')
            )
        except Exception as ex:
            raise SaveError(
                cls=self.name(),
                err=f'failed to persist'
                    f' metadata ({ex}).'
            )
        # save parameters:
        try:
            torch.save(self.state_dict(), parameters_f)
        except Exception as ex:
            raise SaveError(
                cls=self.name(),
                err=f'failed to persist '
                    f'parameters ({ex}).'
            )

    @classmethod
    def load(cls, from_):
        if not os.path.isdir(from_):
            raise LoadError(
                cls=cls.name(),
                err=f'{from_} is not'
                    f' a directory. '
            )

        metadata_f = os.path.join(
              from_,
              cls.metadata_fname()
        )
        parameters_f = os.path.join(
              from_,
              cls.parameters_fname()
        )
        # load metadata:
        try:
            metadata = pickle.load(
             open(metadata_f, 'rb')
            )
        except Exception as ex:
            raise LoadError(
                cls=cls.name(),
                err=f'could not load metadata '
                    f'from {metadata_f} ({ex})'
            )
        # load parameters:
        try:
            parameters = torch.load(parameters_f)
        except Exception as ex:
            raise LoadError(
                cls=cls.name(),
                err=f'could not load parameters '
                    f'from {parameters_f} ({ex})'
            )

        module = cls(*metadata)
        module.load_state_dict(
                    parameters)

        return module
