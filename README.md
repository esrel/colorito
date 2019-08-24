# colorito

Python library for **color normalization and color similarity inference.**

## What Does It Do?

Starting from a color (provided either as text or as an RGB tuple), colorito can be used to:
* **retrieve similar colors**;
* **guess the rgb value of the given color**;
* **guess the main tint** (red, orange, blue, brown, black, pink, ...) of the given color.

colorito can be used to **address underspecification in search constraints**. 

Imagine, for instance, a scenario where a user is searching for a _red_ product in 
your catalog. When retrieving results, you may not want to limit yourself only to 
products whose color is exactly _red_. You could also retrieve products with colors 
that are similar, like _magenta_, _fire red_ and any other different shade you may 
have in your database. 

## Quick Start

To install colorito, run:

`pip install git+https://github.com/kekgle/colorito`

Once installed, try out the snippet below to see how it works. 

```python
from colorito import HTML_PALETTE
from colorito.palette import SmartPalette
from colorito.genies.knn import KNNColorGenie

g = KNNColorGenie()  # create the color genie
p = SmartPalette(g)  # create the smart palette

# add color-name: color-rgb mappings to palette
p.update_palette(HTML_PALETTE)
# prepare the palette (train the color genie to
# perform inference on similar colors)
p.prepare()

# the palette is now ready, you can try getting
# the rgb of a known color, for instance `red`:
print(p.get_color_rgb('red'))

# and of an unknown color (not in HTML_PALETTE):
print(p.get_color_rgb('pale blue'))

# you can also get colors similar to the one we
# provided, even when it's not in HTML_PALETTE:
print(p.get_shades_of('pale blue', as_rgb=False))

# and you can get the main tint:
print(p.get_main_tint('pale blue', as_rgb=False))

```

Additionaly information are provided in the section below 
([how does it work](#how-does-it-work)).

## How Does It Work?

colorito is **designed to be extensible** and **can be improved by 
implementing new [color genies](#colorgenie) for your 
[smart palette](#smartpalette)**


### SmartPalette
The main component of colorito is called **SmartPalette**. A SmartPalette is an
object containing a mapping between color name and rgb value. This mapping needs
to be provided to the SmartPalette upon initialization (as the path to a csv 
file).

The library already provides two large color mappings, that you should augment
with the {_color name_: _color rgb_} pairs that are in your database. They can
be imported with `from colorito import HTML_PALETTE, DEFAULT_PALETTE`.

You can add a color mapping to the SmartPalette with:
```python
>>> from colorito import HTML_PALETTE
>>> from colorito.genies.knn import KNNColorGenie
>>> from colorito.palette import SmartPalette
>>> p = SmartPalette(KNNColorGenie())
>>> p.update_palette(HTML_PALETTE)
```

If you want to add mappings from multiple files, you can call `update_palette` again:
```python
>>> from colorito import DEFAULT_PALETTE
>>> p.update_palette(DEFAULT_PALETTE)
```

### ColorGenie
The color genie is what makes the SmartPalette _smart(er)_. You can implement
your own genies by implementing the abstract class at 
`colorito.genies.ColorGenie`.

For now, only the `KNNColorGenie` is available, which uses the **k-nearest
neighbour algorithm to retrieve the most similar colors**. The distances
between colors are computed using the _delta E 2000_ score (which is more
perceptually accurate than simple Euclidean distance on the RGB space).

The `KNNColorGenie` is very simple and **weak at inferring the rgb of a color
that it is not in the palette**. In fact, all it does is featurize the names
of the colors in the palette (using a bag-of-words approach) and then, given
the name of a new color, it returns the rgb of the color in the palette with
the highest cosine similarity.

This is obviously sub-optimal as there is no real _understanding_ of the
semantics in colors' names. For instance, if `light red` and `dark red` are
known and we ask the ColorGenie what is the most likely rgb for `pale red`,
the `KNNColorGenie` will assign equal probability to `light red` and `dark 
red` as they both have the word `red` in common with `pale red`. The genie
will not capture the higher similarity between `pale` and `light`.

**Smarter approaches** that include language understanding will be
**implemented at a later time**.
