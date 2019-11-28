# colorito

Python library for **natural language color search.**

## What Does It Do?

Given an initial list of color names, colorito is capable of **retrieving**, from 
said list, all **colors that are perceptually similar** to a specified one.

## Installation

To install colorito, run:

`pip install https://github.com/kekgle/colorito`


## Quick Start

The main component in colorito is called **SmartPalette**. A SmartPalette is 
built starting from a list of color names. These names are then transformed 
into real colors (`colorito.colors.Color`), by a color generator (namely, a 
neural network). 

```python
>>> from colorito.palette import SmartPalette
>>> palette = SmartPalette(colors=color_list)
```

You can provide the list of colors as either a Python List, or as the path
to a text file, where each line of the file contains a color name. If you
don't specify a list of colors, a default one is used.

A **SmartPalette provides two main APIs**:

* `search` - given a color name, it **returns colors similar to the specified 
one**; the returned colors are retrieved from the list that was used to 
initialize the SmartPalette.

* `invent` - given a color name, it generates a `colorito.colors.Color` object, 
which includes the color's RGB value and CIELab value. The color can be 
visualized by calling its `render()` method.

### SmartPalette.search()

```python
>>> from colorito.palette import SmartPalette
>>> p = SmartPalette()
>>> colors, simil_scores = p.search('water')
>>> colors[0].name
INFO:colorito:data:utils: cleaning strings...
'Blue Diamond'
```

By default, `search()` sets a similarity threshold based on the elbow method
(by finding the point of maximum curvature in the similarity curve between
the searched color and those provided upon initialization of the 
SmartPalette). 

However, You can manually limit search results by passing additional keyword 
arguments to `search()`:

* `search(color_name, n=10)` - returns only the top 10 colors that are the
most similar to `color_name`;
* `search(color_name, t=.5)` - returns only colors that have a similarity
greater than 0.5 (50%) with respect to `color_name`;

When using plain `search(color_name)`, the simil

### SmartPalette.invent()

```python
>>> from colorito.palette import SmartPalette
>>> p = SmartPalette()
>>> color = p.invent('pink')
>>> color.rgb
INFO:colorito:data:utils: cleaning strings...
(207, 130, 161)
>>> color.hexc
INFO:colorito:data:utils: cleaning strings...
'#cf82a1'
>>> color.lab
INFO:colorito:data:utils: cleaning strings...
(63.18214860087497, 33.85211035054453, -3.969598637729832)
```

Additional information are provided in the section below 
([how does it work](#how-does-it-work)).

## How Does It Work?

The core of colorito is a **neural model** (developed using [PyTorch](https://www.pytorch.org)),
that was trained on **generating color coordinates from color names**. 

### Training
The network used by colorito was trained using **almost 40k pairs** of color names - hexadecimal 
values, that were scraped from different web sources (scraping notebooks are available in
the [notebooks](https://www.github.com/kekgle/colorito/) folder).

During training, the **network learns to minimize the MSE of the generated CIELab coordinates of
a color**, starting from its name (the coordinates are normalized in the [0, 1] range). 
The choice of working with the CIELab space rather than RGB, is due to the fact that, in the 
former, the Euclidean Distance between colors translates better to their perceptual distance 
(compared to the RGB space).

After the network has been trained, it is used to generate 256-dimensional embeddings of colors'
names, that are "perceptually aware" (meaning that names representing similar colors, will have
similar embeddings).


### Models

As of now, there are two available color generators: 

* one uses an LSTM encoder with 512 hidden unit, two layers, followed by a 512-dimensional and a 
256-dimensional dense layers with ReLU and tanh activations respectively; this network uses unigrams,
bigrams and trigrams feature at the character level.

* the other (dubbed Lite due to its reduced size), is composed of a 256-unit LSTM encoder, with two
layers, that uses only character unigram features, followed by the same dense layers used by the
"beefier" model.

You can find a notebook where training of both networks can be reproduced in the
[notebooks](https://www.github.com/kekgle/colorito/) folder.

