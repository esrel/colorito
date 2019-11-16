from colorito.nnet.modules import SmartModule
from colorito.nnet.modules.lstm import RecurrentModule
from colorito.nnet.modules.lite import LiteRecurrentModule as LiteRecurModule
from colorito.nnet.modules.lin import LinearModule
from colorito.utils.fs import mkdir
from colorito.utils.logs import setup_logger
from colorito.exceptions import SaveError, LoadError

import torch.nn as nn
import torch
import pickle
import os

logger = setup_logger('color-generator')


class ColorGenerator(nn.Module):

    # contains SmartModules that are usable as
    # sub-modules for a ColorGenerator; new mo-
    # dules have to be added to this registry,
    # before being usable.

    REGISTRY = {
        'rnn': {
            RecurrentModule.name(): RecurrentModule,
            LiteRecurModule.name(): LiteRecurModule
        },
        'lin': {
            LinearModule.name(): LinearModule
        }
    }

    def __init__(self, rnn, lin):
        self._validate_registry()
        assert isinstance(rnn, SmartModule), f'{rnn.__class__.__name__} is not a SmartModule'
        assert isinstance(lin, SmartModule), f'{lin.__class__.__name__} is not a SmartModule'
        assert (
            rnn.name() in self.REGISTRY['rnn']
        ), (
            f'{rnn.name()} is not registered as a '
            f'valid rnn module - you can register '
            f'a module by adding it to the ColorGe'
            f'nerator\'s REGISTRY class variable. '
        )
        assert (
            lin.name() in self.REGISTRY['lin']
        ), (
            f'{lin.name()} is not registered as a '
            f'valid lin module - you can register '
            f'a module by adding it to the ColorGe'
            f'nerator\'s REGISTRY class variable. '
        )

        super().__init__()
        self.lin = lin
        self.rnn = rnn

    def _validate_registry(self):
        """
        Validates the registry by making sure that there
        are no classes shared between different types of
        sub-modules.

        :return:
        """
        classes = [
            name for name in {
                _ for d in self.REGISTRY.values() for _ in d.keys()
            }
        ]
        assert (
            len(classes) == len(set(classes))
        ), (
            'Registry is invalid. Some class '
            'is registered more than once as '
            'a different sub-component.'
        )

    def forward(self, x):
        x, _ = self.rnn(x)
        x, _ = self.lin(x)
        return x

    def h(self, x):
        self.eval()
        with torch.no_grad():
            x, _ = self.rnn(x)
            _, h = self.lin(x)

        self.train()

        return h

    def y(self, x):
        self.eval()
        y = self(x)

        self.train()

        return y

    def save(self, to):
        """
        Saves the ColorGenerator to file-system.

        :param to:
        :return:
        """
        module = self.__class__.__name__
        if not os.path.isdir(to):
            raise SaveError(
                cls=module ,
                err=f'{to} is not '
                    f'a directory.'
            )

        # save metadata (contains class names of
        # the sub-modules used by the generator)

        rec_module = self.rnn.__class__.__name__
        lin_module = self.lin.__class__.__name__
        metadata = {
            'rnn': rec_module,
            'lin': lin_module
        }

        metadata_f = os.path.join(to, self.metadata_fname())
        try:
            pickle.dump( metadata,  open(metadata_f, 'wb') )
        except Exception as ex:
            raise SaveError(
                cls=module ,
                err=f'unable to save metadata'
                    f' to {metadata_f} ({ex})'
            )

        # save sub-modules to file-system:

        modules = os.path.join(to, 'modules')
        mkdir(modules)

        def _save_submodule(submodule, dest):
            mkdir(dest), submodule.save(dest)

        _save_submodule(self.rnn, os.path.join(modules, 'rnn'))
        _save_submodule(self.lin, os.path.join(modules, 'lin'))

    @classmethod
    def load(cls, from_):
        """
        Loads a ColorGenerator from file-system.

        :param from_:
        :return:
        """
        module = cls.__name__
        if not os.path.isdir(from_):
            raise LoadError(
                cls=module ,
                err=f'{from_} is not'
                    f' a directory. '
            )

        # load metadata first to discov-
        # er classes of the sub-modules:

        metadata_f = os.path.join(from_, cls.metadata_fname())
        try:
            metadata = pickle.load( open( metadata_f, 'rb' ) )
        except Exception as ex:
            raise LoadError(
                cls=module ,
                err=f'could not load metadata'
                    f' from: {from_} ({ex}). '
            )

        # ensure classes in metadata are valid:

        if not (
            metadata.get('rnn') and
            metadata.get('lin')
        ):
            raise LoadError(
                cls=module ,
                err='invalid metadata (missing'
                    ' `rnn` and/or `lin` keys)'
            )
        try:
            rnn = cls.REGISTRY['rnn'][metadata['rnn']]
            lin = cls.REGISTRY['lin'][metadata['lin']]

        except KeyError:
            raise LoadError(
                cls=module ,
                err='invalid metadata (invalid'
                    ' `rnn` and/or `lin` keys)'
            )

        modules = os.path.join(from_, 'modules')
        # load sub-modules:
        rnn = rnn.load(os.path.join(modules, 'rnn'))
        lin = lin.load(os.path.join(modules, 'lin'))

        return cls(rnn, lin)

    @classmethod
    def metadata_fname(cls):
        return f'{cls.__name__}.mdata.pl'
