from colorito.nnet.modules import SmartModule
from colorito.nnet.modules.encoders.lstm import LSTMEncoder, LiteEncoder
from colorito.nnet.modules.decoder import Decoder
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
        'encoder': {
            LSTMEncoder.name(): LSTMEncoder,
            LiteEncoder.name(): LiteEncoder
        }
    }

    def __init__(self, encoder):
        assert isinstance(encoder, SmartModule), f'{encoder.__class__.__name__} is not a SmartModule'
        assert (
            encoder.name() in self.REGISTRY['encoder']
        ), (
            f'{encoder.name()} is not registered as a '
            f'valid encoder module - you can register '
            f'a module by adding it to the ColorGenera'
            f'tor\'s REGISTRY class variable. '
        )
        super().__init__()
        self.decoder = Decoder(encoder.output_size())
        self.encoder = encoder

    def forward(self, x):
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)
        return x

    def h(self, x):
        self.eval()
        with torch.no_grad():
            x, _ = self.encoder(x)
            _, h = self.decoder(x)

        self.train(mode=True)

        return h

    def y(self, x):
        self.eval()
        y = self(x)

        self.train(mode=True)

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

        encoder_module = self.encoder.__class__.__name__
        decoder_module = self.decoder.__class__.__name__
        metadata = {
            'encoder': encoder_module,
            'decoder': decoder_module
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

        _save_submodule(self.encoder, os.path.join(modules, 'encoder'))
        _save_submodule(self.decoder, os.path.join(modules, 'decoder'))

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
            metadata.get('encoder') and
            metadata.get('decoder')
        ):
            raise LoadError(
                cls=module ,
                err='invalid metadata (missing'
                    ' `encoder` and/or `decoder` keys)'
            )
        try:
            encoder = cls.REGISTRY['encoder'][metadata['encoder']]

        except KeyError:
            raise LoadError(
                cls=module ,
                err='invalid metadata (invalid `en'
                    'coder` and/or `decoder` keys)'
            )

        modules = os.path.join(from_, 'modules')
        # load sub-modules:
        encoder = encoder.load(os.path.join(modules, 'encoder'))
        decoder = Decoder.load(os.path.join(modules, 'decoder'))

        return cls(encoder, decoder)

    @classmethod
    def metadata_fname(cls):
        return f'{cls.__name__}.mdata.pl'
