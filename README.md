# Harmony Align

This a submodule to facilitate the alignment of a set of atom coordinates to either:
* a 2D class, representing the projected atom density. Typically given as a frame in an `.mrcs` file.
* a volume map, typically given as a `.vol` of `.mrcs` file.

The main requirements for this module are:
* it must be a global optimization
* the coordinate fitting mus be extremely fast, at the expense of preprocessing the densities at will

These come from the need to integrate it in Boltz diffusion pipeline, where coordinates can come in any orientation and are reproduced many times (default 200) for the same densities against which we want to validate.

## Correlation approximation

A typical way to compare 2 densities is to take the correlation between its pixels. A proven benefical approach is to reduce the densities pixel content to orthogonal components carrying the bulk of the information, reducing the computational cost. This has been done in Xmipp both by PCA and other decompositions. Representing the reference density for a large number of candidate rotations by such a reduced vector could allow fast comparison to a new set of coordinates, if we can convert them efficiently to the same basis.

A workflow to this end could like this:
```python
from tools.spider_files3 import open_volume

# preprocess
reference_density = open_volume('reference.vol')
gallery: list[tuple[int,int,int]] = create_gallery(angular_resolution)
rotated_densities = {(psi,tilt,rot):rotate_volume(density,rot) for rot in gallery}
rotated_components = {rot: encode_density(density) for rot, density in rotated_densities.items()}
components = np.stack(rotated_components.values())

# diffusion process starts

# coordinate-step
gallery_idx = np.argmax(components @ encode_coordinates(diffusion_coordinates))
```

## Spherical harmonics

An additional computational benefit comes from a decomposition in orthogonal components that are also in a representation of the rotation group, allowing very efficient rotation through a linear transformation. This means that angular part of the decomposition must be the spherical harmonics.

For matching spherical distributions (so without radial part) this decomposition is called the Spherical Fourier Transform (SFT) and bot for its own implementation as the subsequent matching step very efficient algorithms exist, both in a Machine Learning context and astronomy, as far as I could find. It has also been proposed for use with molecular data (originally from tomography), e.g.by [Friedman in 1999](https://www.sciencedirect.com/science/article/pii/S0097848598000266?ref=pdf_download&fr=RR-2&rr=95d0a76909a8b9c0).

If a radial component is present, there is a bit more freedom, since we have "easy transformation" constraint on that. The most general approximation is the Fourier-Bessel expansion:

$$V = \int dk\sum_{lm} v_{lm}(k) j_l(kr) Y_{lm}(\theta,\phi)$$

Which has *a lot* for terms given we need to integrate over all frequencies `k`

Another expansion would be the Zernike fourier transform

$$V = \sum_{nlm} X_{nlm} \frac{j_{2n+l+1}(r)}{r} Y_{lm}(\theta,\phi)$$

for densities with a bounded frequency domain.

In either of these expansions the crucial step is to be able to identify the relevant terms, based on the reference density, and reduce our encoding to those. 

## Heat Kernel

In the previous discussion it has been assumed the encoding stems from an approximation of the density in all (voxels of) 3D space. However, one of our proposed encodings is ment to work with coordinates instead. 

Typically we would make the conversion to a smooth density by applying a Gaussian convolution over the discrete atom coordinates:

$$V \sim \sum_a e^{-\frac{(\vec x - \vec x_a)^2}{2\sigma^2}}=\sum_a K(\vec x,\vec x_a,\frac{\sigma^2}{2})$$

but it is interesting to consider this in another way: The heat distribution of a number isolated hot sources will hold the same distribution after a time $\frac{\sigma^2}{2}$. In this context, the Gaussian above is called the _heat kernel_ and it gives in fact a separable expansion of a "metric" connecting our dense volume and isolated coordinates:

$$K(\vec x,\vec y,\frac{\sigma^2}{2}) = \int dk\ e^{-\frac{\sigma^2}{2}k^2} \bigl(k\ j_l(kx)\ Y_{lm}(\Omega_x)\bigr)\bigl(k\ j_l(ky)\ Y^*_{lm}(\Omega_y) \bigr)$$

This exprssion allow to contract the `x` side with dense coordinates and the `y` side with discrete ones and interpret the result as `x` contracted with the Gaussian heat-diffused evolution of `y`, exactly what we need. 

It is of course possible to do this very same derivation from fourier transforming the Gaussian without ever considering the heat equation, but it is perhaps more interesting to keep this underlying principle in mind.

## Volume to projection

Interestingly, solving the problem for comparing coordinates to volumes trivially solves it for 2D densities that we can smear out over the third dimension. Indeed, correlating a 2D density with sum of a 3D density over the third dimension is no different than correlating such a smeared out 2D density, constant in the third direction, with the full 3D density.

Conversely, we expect that the preprocessing of the 2D case will be much more efficient, since the conversion to 3D clearly adds extra computational cost. However, the diffusion step only encodes coordinates and contracts encoded vectors, for which both cases should have an identical procedure.

## Implementation

Currently the latter approach, selecting relevant components in the heat kernel approximation, is implemented both for CPU (numpy) and GPU (torch) execution in `align/volume_heat_kernel` and `align/volume_heat_kernel_torch` respectively. Perhaps surprisingly, the diffusion computation doesn't differ much between them with torch being even slighly slower, perhaps the missing bessel and spherical functions being interpolated instead can account for that. The preprocessing step on the other hand is much faster on the GPU.

# Getting started

Testing requires Xmipp to be present, for creating the volume densities from a pdb, but further there are only python dependencies that should be installed with 

```
pip install -e .
```
preferably in a dedicated virtual environment.

The primary test script can then be called as 
```
python src/main.py
```
and should return an estimate of a given rotation.
