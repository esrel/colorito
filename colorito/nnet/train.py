from colorito import COLORS, DEVICE, MODELS

from colorito.utils.logs import setup_logger

from colorito.data.vectorize import NgramVectorizer
from colorito.data.dataset import ColorDataset

from colorito.nnet.model import ColorGenerator
from colorito.nnet.modules.encoders.lstm import LSTMEncoder, LiteEncoder

from torch.utils.data.dataloader import DataLoader
from datetime import datetime
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import numpy as np
import argparse
import os


defaults = {
    'data': COLORS,
    'output': MODELS,
    'ngrams': 3,
    'epochs': 5,
    'batch_size': 32,
    'learning_rate': 1e-03,
    'decay': 0.0
}

logger = setup_logger('color-generator:train')


def train(
    data=defaults['data'],
    output=defaults['output'],
    ngrams=defaults['ngrams'],
    epochs=defaults['epochs'],
    batch_size=defaults['batch_size'],
    learning_rate=defaults['learning_rate'],
    decay=defaults['decay'],
    lite=True,
    wbound=False
):
    """
    Trains a neural network to generate colors from text data;
    namely, a ColorGenerator.

    :param data:
    :param output:
    :param ngrams:
    :param epochs:
    :param batch_size:
    :param learning_rate:
    :param decay:
    :param lite:
    :param wbound:
    :return:
    """

    logger.info(
        f' will train a {"lite " if lite else ""}ColorGenerator.'
    )

    if not os.path.isdir(output):
        raise ValueError(
            f' Invalid output path: {output} is not a directory.'
        )

    logger.info(
        f' building vectorizer with max n-gram order of {ngrams} '
        f'(n-grams with{"out" if not wbound else ""} word bounds)'
    )

    vectorz = NgramVectorizer(ngrams, bound=wbound)
    dataset = ColorDataset.build(
        data,
        vectorizer=vectorz,
        space='lab'
    )

    logger.info(f' assembling the network...')

    input_dim = tuple(dataset.x[0].size())
    if lite:
        encoder = LiteEncoder(
            input_dim=input_dim,
            lexicons_=vectorz.lexicons,
            ngram_order=ngrams,
            ret_sequences=False
        )
    else:
        encoder = LSTMEncoder(
            input_dim=input_dim,
            lexicons_=vectorz.lexicons,
            max_ngram_order=ngrams,
            ret_sequences=False
        )

    cg = ColorGenerator(encoder)

    logger.info(
        f' will train for {epochs} epochs; with {batch_size}-sized '
        f'mini-batches and with a learning rate of {learning_rate} '
        + '' if not decay else f'(decaying by {decay} per epoch...)'
    )

    criterion_ = nn.MSELoss()
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=True
    )

    # train the model

    for epoch in range(epochs):

        lr = learning_rate - epoch * decay
        optimizer = torch.optim.Adam(
            cg.parameters(),
            lr=lr
        )

        avg_epoch_loss, loss = [], None

        logger.info(
            f' Epoch {epoch + 1}/{epochs} will '
            f'run with a learning rate of: {lr}'
        )
        pbar = tqdm(
            total=round(len(dataset) / batch_size)
        )

        for batch in dataloader:
            X, y = batch
            X.to(DEVICE)

            # zero gradient

            optimizer.zero_grad()

            # forward-pass

            y_hat = cg(X)

            # compute loss

            loss = criterion_(y_hat, y)

            # back-prop

            loss.backward()

            pbar.set_description(f' loss: {loss.item()}')
            avg_epoch_loss.append(loss.item())

            optimizer.step()

            pbar.update(1)

        avg_epoch_loss = np.mean(avg_epoch_loss)
        logger.info(f' Average Loss: {avg_epoch_loss}')
        pbar.close()

    # save the model!

    output = os.path.join(output, 'color-generator')
    if os.path.isdir(output):
        # append timestamp to avoid clashes
        output += f' ({datetime.utcnow()})'

    logger.info(f' will save model to: {output}...')

    os.mkdir(output)
    cg.save (output)


def argument_parser():

    parser = argparse.ArgumentParser(
        description='Colorito command line: '
                    'train a ColorGenerator!'
    )
    parser.add_argument(
        '-d',
        '--data',
        default=defaults['data'],
        help='Path to the folder containing '
             'the csv with colors. Each file'
             ' should have two columns: one '
             'with the names, and the other '
             'with the hexadecimal rgb value'
    )
    parser.add_argument(
        '-o',
        '--output',
        default=defaults['output'],
        help='Path to where the model is saved'
    )
    parser.add_argument(
        '-n',
        '--ngrams',
        default=defaults['ngrams'],
        type=int,
        help='Maximum order of n-grams that are'
             ' extracted from the training data'
    )
    parser.add_argument(
        '--wbound',
        action='store_true',
        help='Add word-bounds to n-gram features'
    )
    parser.add_argument(
        '--lite',
        action='store_true',
        help='Train a Lite ColorGenerator. The'
             ' Lite version of ColorGenerator '
             'has less parameters and will use'
             ' only max-order ngrams as feats '
             '(trains typically 5x faster)'
    )
    parser.add_argument(
        '-e',
        '--epochs',
        default=defaults['epochs'],
        type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        default=defaults['batch_size'],
        type=int,
        help='Size of a training batch'
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        default=defaults['learning_rate'],
        type=float,
        help='Learning rate for optimizing'
    )
    parser.add_argument(
        '--decay',
        default=defaults['decay'],
        type=float,
        help='Learning rate decay per epoch'
    )

    return parser


if __name__ == '__main__':

    args = argument_parser().parse_args()

    train(**vars(args))
