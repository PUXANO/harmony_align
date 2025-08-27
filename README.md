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

Currently the latter approach, selecting relevant components in the heat kernel approximation, is implemented in `align/volume_heat_kernel`. A previous `torch` implementation is omitted since it was quite complex to implement the special functions there. The speedup was significant in the preprocessing, but we can replicate this by parallelizing over multiple cpu's. It is not worth maintaining a dual implementation in the end.

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

# Benchmark

We evaluated the current implementation on a cluster with 13 proteines from the EMDB, aligning their pdb to the volume map.

The preprocessing into Fourier-Bessel moments was parallelized over 30 CPU cores on the Karolina cluster. Radial eigenvalues were taken as the first 50 corresponding bessel roots, while the maximal angular eigenvalue was $l=8$. All volume maps were rescaled to have box size between 100 and 200 voxels, to avoid arrays being to large to dispatch to other processes. Finally, we (ab)used prior knowledge of the relative translation between pdb and volume map to only test the matching of the rotational degrees of freedom here.

Results are:
|    | emdb_id   |   n_atoms |   box_size |   psi_in |   tilt_in |   rot_in |   psi_out |   tilt_out |   rot_out |   deviation |   align_time |   prep_time |
|---:|:----------|----------:|-----------:|---------:|----------:|---------:|----------:|-----------:|----------:|------------:|-------------:|------------:|
|  🟩 | EMD-30229 |      8824 |        160 |      -83 |        35 |      129 |       -80 |         36 |       130 |     4.43353 |     2.4125   |     78.7917 |
|  🟩 | EMD-30229 |      8824 |        160 |     -161 |        21 |      -71 |      -166 |         21 |       -65 |     2.27651 |     2.39582  |     78.7917 |
|  🟩 | EMD-30229 |      8824 |        160 |      -94 |        95 |     -176 |       -95 |         98 |      -175 |     2.97938 |     2.40028  |     78.7917 |
|  🟩 | EMD-30229 |      8824 |        160 |       79 |        93 |       65 |        80 |         93 |        65 |     1.27306 |     2.39552  |     78.7917 |
|  🟩 | EMD-30229 |      8824 |        160 |      -74 |        65 |      140 |       -73 |         67 |       140 |     3.03272 |     2.39446  |     78.7917 |
|  🟩 | EMD-37725 |      9892 |        120 |       79 |        92 |      -84 |        80 |         93 |       -85 |     1.87053 |     2.11154  |     27.9261 |
|  🟩 | EMD-37725 |      9892 |        120 |        7 |        49 |     -150 |         3 |         51 |      -150 |     3.81082 |     2.11114  |     27.9261 |
|  🟩 | EMD-37725 |      9892 |        120 |       98 |        38 |     -134 |        96 |         36 |      -135 |     3.18481 |     2.11184  |     27.9261 |
|  🟩 | EMD-37725 |      9892 |        120 |      -35 |        71 |      -33 |       -34 |         72 |       -35 |     1.79185 |     2.10533  |     27.9261 |
|  🟩 | EMD-37725 |      9892 |        120 |      105 |       111 |       92 |       105 |        113 |        90 |     2.97559 |     2.1085   |     27.9261 |
|  🟩 | EMD-42892 |      8874 |        144 |      124 |       134 |      -82 |       126 |        134 |       -80 |     1.50299 |     1.23487  |     56.2216 |
|  🟩 | EMD-42892 |      8874 |        144 |       92 |       110 |     -140 |        91 |        108 |      -140 |     2.19424 |     1.23462  |     56.2216 |
|  🟩 | EMD-42892 |      8874 |        144 |      -29 |       149 |      -51 |       -34 |        149 |       -55 |     2.64845 |     1.23422  |     56.2216 |
|  🟩 | EMD-42892 |      8874 |        144 |       96 |        67 |        0 |        94 |         67 |         0 |     1.8029  |     1.23431  |     56.2216 |
|  🟩 | EMD-42892 |      8874 |        144 |      129 |        82 |      -91 |       130 |         82 |       -90 |     1.93425 |     1.23131  |     56.2216 |
|  🟩 | EMD-19016 |      7404 |        100 |      -26 |        35 |      112 |       -29 |         36 |       115 |     2.26496 |     0.664462 |     16.4148 |
|  🟩 | EMD-19016 |      7404 |        100 |      -35 |       115 |       27 |       -35 |        113 |        25 |     2.67352 |     0.659776 |     16.4148 |
|  🟩 | EMD-19016 |      7404 |        100 |     -138 |        50 |      164 |      -136 |         51 |       165 |     2.81137 |     0.661057 |     16.4148 |
|  🟩 | EMD-19016 |      7404 |        100 |      -71 |        59 |      -74 |       -67 |         62 |       -75 |     4.10348 |     0.662304 |     16.4148 |
|  🟩 | EMD-19016 |      7404 |        100 |       14 |        18 |      -19 |        18 |         15 |       -20 |     4.09814 |     0.661993 |     16.4148 |
|  🟩 | EMD-16264 |     38177 |        100 |      141 |        30 |       78 |       141 |         31 |        75 |     2.73796 |     0.94444  |     17.7441 |
|  🟨 | EMD-16264 |     38177 |        100 |     -168 |        89 |      -52 |      -165 |         93 |       -50 |     5.06252 |     0.940281 |     17.7441 |
|  🟨 | EMD-16264 |     38177 |        100 |       52 |        74 |      -40 |        55 |         72 |       -45 |     5.44395 |     0.939975 |     17.7441 |
|  🟩 | EMD-16264 |     38177 |        100 |      -50 |        54 |       80 |       -54 |         51 |        85 |     4.88988 |     0.943731 |     17.7441 |
|  🟩 | EMD-16264 |     38177 |        100 |      130 |       119 |     -133 |       129 |        118 |      -135 |     2.11348 |     0.942857 |     17.7441 |
|  🟨 | EMD-41784 |      6634 |        160 |      -50 |       119 |       97 |       -50 |        123 |       100 |     5.67149 |     1.96868  |     83.6229 |
|  🟨 | EMD-41784 |      6634 |        160 |      106 |        26 |       63 |       111 |         21 |        55 |     6.84676 |     1.96794  |     83.6229 |
|  🟨 | EMD-41784 |      6634 |        160 |      163 |        16 |       15 |      -180 |         15 |        -5 |     6.10167 |     1.96348  |     83.6229 |
|  🟩 | EMD-41784 |      6634 |        160 |      -34 |       163 |      -31 |       -36 |        165 |       -30 |     3.35252 |     1.96527  |     83.6229 |
|  🟨 | EMD-41784 |      6634 |        160 |     -125 |        81 |     -122 |      -130 |         82 |      -120 |     5.14235 |     1.96804  |     83.6229 |
|  🟩 | EMD-27750 |      9312 |        128 |       -4 |        63 |      -86 |        -6 |         62 |       -85 |     1.69903 |     2.54742  |     37.1535 |
|  🟩 | EMD-27750 |      9312 |        128 |     -135 |       127 |     -105 |      -136 |        129 |      -105 |     2.1868  |     2.54837  |     37.1535 |
|  🟩 | EMD-27750 |      9312 |        128 |      104 |        37 |       81 |       105 |         36 |        80 |     1.00081 |     2.54865  |     37.1535 |
|  🟩 | EMD-27750 |      9312 |        128 |       35 |        66 |      -97 |        35 |         67 |       -95 |     1.93831 |     2.54745  |     37.1535 |
|  🟩 | EMD-27750 |      9312 |        128 |     -101 |        60 |       34 |      -101 |         62 |        35 |     2.14049 |     2.5466   |     37.1535 |
|  🟨 | EMD-29965 |      6110 |        100 |      -93 |        79 |      -74 |       -91 |         72 |       -80 |     9.14571 |     0.262742 |     17.7371 |
|  🟨 | EMD-29965 |      6110 |        100 |     -155 |        49 |     -118 |      -158 |         41 |      -115 |     8.2019  |     0.257975 |     17.7371 |
|  🟨 | EMD-29965 |      6110 |        100 |      -24 |       109 |       96 |       -24 |        113 |        90 |     7.52233 |     0.257658 |     17.7371 |
|  🟨 | EMD-29965 |      6110 |        100 |     -131 |        86 |       87 |      -129 |         77 |        85 |     9.02607 |     0.258156 |     17.7371 |
|  🟨 | EMD-29965 |      6110 |        100 |     -137 |       110 |       15 |      -134 |        103 |        15 |     7.08564 |     0.258821 |     17.7371 |
|  🟥 | EMD-31676 |      8997 |        128 |      176 |        83 |       74 |        72 |        165 |       -35 |   102.497   |     0.487834 |     33.8739 |
|  🟥 | EMD-31676 |      8997 |        128 |      -43 |        54 |       90 |      -137 |         67 |      -145 |   101.137   |     0.484928 |     33.8739 |
|  🟥 | EMD-31676 |      8997 |        128 |       29 |        67 |       66 |       150 |         41 |       -80 |   101.537   |     0.484459 |     33.8739 |
|  🟥 | EMD-31676 |      8997 |        128 |       31 |        44 |        6 |       163 |         62 |      -150 |   102.567   |     0.484939 |     33.8739 |
|  🟥 | EMD-31676 |      8997 |        128 |       -6 |        56 |       -5 |      -166 |         46 |       155 |   100.685   |     0.484367 |     33.8739 |
|  🟩 | EMD-27265 |      5730 |        144 |      150 |        84 |       78 |       150 |         82 |        80 |     2.67124 |     0.636259 |     52.0537 |
|  🟩 | EMD-27265 |      5730 |        144 |      116 |       115 |       81 |       116 |        113 |        80 |     2.04777 |     0.63465  |     52.0537 |
|  🟩 | EMD-27265 |      5730 |        144 |      123 |       122 |     -178 |       127 |        123 |      -175 |     3.72227 |     0.636291 |     52.0537 |
|  🟩 | EMD-27265 |      5730 |        144 |     -146 |       143 |     -166 |      -147 |        144 |      -165 |     1.36113 |     0.636769 |     52.0537 |
|  🟩 | EMD-27265 |      5730 |        144 |     -103 |        76 |       97 |      -104 |         77 |        95 |     2.66812 |     0.638472 |     52.0537 |
|  🟩 | EMD-45632 |     11455 |        128 |       13 |        48 |       78 |        17 |         46 |        75 |     3.0901  |     1.91006  |     36.7283 |
|  🟩 | EMD-45632 |     11455 |        128 |      169 |       133 |      -79 |       173 |        134 |       -75 |     3.32721 |     1.9046   |     36.7283 |
|  🟩 | EMD-45632 |     11455 |        128 |       19 |       144 |       34 |        21 |        144 |        35 |     1.37268 |     1.90684  |     36.7283 |
|  🟩 | EMD-45632 |     11455 |        128 |      -78 |       109 |       33 |       -76 |        108 |        35 |     2.8996  |     1.91194  |     36.7283 |
|  🟩 | EMD-45632 |     11455 |        128 |       27 |       117 |      173 |        28 |        118 |       170 |     3.63963 |     1.90521  |     36.7283 |
|  🟩 | EMD-47944 |      4000 |        170 |       28 |       119 |      -74 |        28 |        118 |       -75 |     1.11622 |     1.63743  |    102.148  |
|  🟩 | EMD-47944 |      4000 |        170 |     -160 |        95 |      153 |      -160 |         98 |       150 |     3.90348 |     1.63156  |    102.148  |
|  🟩 | EMD-47944 |      4000 |        170 |      -27 |       105 |     -120 |       -23 |        103 |      -115 |     5.96682 |     1.6364   |    102.148  |
|  🟩 | EMD-47944 |      4000 |        170 |     -102 |        33 |     -110 |      -105 |         36 |      -105 |     5.16882 |     1.63446  |    102.148  |
|  🟩 | EMD-47944 |      4000 |        170 |       47 |       124 |     -123 |        50 |        123 |      -120 |     2.88288 |     1.63596  |    102.148  |
|  🟩 | EMD-22687 |     18330 |        128 |      128 |        77 |      -68 |       129 |         77 |       -70 |     2.00153 |     3.82174  |     33.9848 |
|  🟩 | EMD-22687 |     18330 |        128 |      -38 |       114 |      110 |       -40 |        113 |       110 |     1.99773 |     3.8188   |     33.9848 |
|  🟩 | EMD-22687 |     18330 |        128 |     -131 |        99 |       23 |      -130 |         98 |        25 |     2.16395 |     3.81955  |     33.9848 |
|  🟩 | EMD-22687 |     18330 |        128 |     -172 |       112 |      -44 |      -169 |        113 |       -40 |     3.72985 |     3.81683  |     33.9848 |
|  🟩 | EMD-22687 |     18330 |        128 |      -89 |       117 |      149 |       -89 |        113 |       150 |     3.77955 |     3.82414  |     33.9848 |

Given the resolution of our search is only 5 degrees, these results perfectly predict all angles for 9-10 of the 13 examples, and a really good match for 2-3 more. Only EMD-31676 has a significant, but consistent, error in the predictions: The pdb seems to match the volume better in a particular, non-trivial rotated state. Inspecting this in detail, one can see the pdb indeed ony explains part of the map.