# Results section to paste into the gdoc

## Observers match fractal dimension to a percept family's spatial scale.

To complement the seed-based taxonomy of Fig. 2 (human vs animal share)
with a finer-grained, scale-oriented view, we hand-curated eight
hierarchically organised percept families spanning the most frequent
vocabulary in the corpus: face / body part, person / people, animal,
mythical creature, plant, place / landscape, object / symbol, and
abstract / texture. Coverage on the corpus of valid percept tokens was
59% (the residual is largely geography terms, foods, and rare animals
unique to a single observer); per-FD report rates within the covered
fraction were z-scored across the three FD conditions to expose the
shape of each family's tuning.

Two converging patterns emerge (Fig. X). First, families differ
sharply in *where* on the FD axis they peak. Faces, body parts,
persons, places, and discrete objects are all over-reported on the
smoothest stimuli (FD ≈ 1.3; z = +0.7 to +1.4) and under-reported on
the busiest ones (FD ≈ 1.7; z = −0.9 to −1.4). Animals peak at the
middle complexity level (FD ≈ 1.5; z = +0.8), as do mythical
creatures (z = +0.4 at FD 1.5 and z = +0.9 at FD 1.7). Plants and
abstract or texture descriptors peak strongly on the busiest stimuli
(FD ≈ 1.7; z = +1.4 in both cases). Second, computing each family's
rate-weighted *preferred FD* (a continuous summary of its tuning)
orders the families on a smooth-to-rough axis from approximately 1.45
(persons, places, faces, objects) through 1.49 (animals) to 1.54
(mythical creatures, abstract textures, plants), recovering the
expected progression from global, coherent forms at low complexity to
fine, repeated structure at high complexity.

The same gradient is visible at the percept level (Fig. X, panel b):
the 99 individual words reported on at least 30 trials span a
preferred-FD range of roughly 1.31 to 1.62, with the family means
falling along the heatmap-derived ordering. The result formalises the
qualitative content shift documented in Fig. 2 as a continuous mapping
between stimulus complexity and the spatial scale of the meanings
observers extract.

---

## Methods paragraph

**Percept-family taxonomy and preferred FD.** We assigned every valid
percept word to one of eight hand-curated semantic families (face /
body part, person / people, animal, mythical creature, plant,
place / landscape, object / symbol, abstract / texture); see
`analysis/figures/fd_family.py` for the full word lists. Words not in
any family contributed to the cross-FD denominator but not to any
per-family numerator (overall family coverage 59% of valid tokens).
For each (family, FD) pair we computed the family's share of all valid
tokens at that FD; rows of the resulting 8 × 3 matrix were z-scored
across FD per family so each row sums to zero and a positive value
identifies the FD at which that family over-reports. A continuous
*preferred FD* per family was computed as the rate-weighted mean of the
three measured fractal dimensions
(FD12: 1.27, FD14: 1.49, FD16: 1.67;
see Materials and Methods, *Stimuli*); a per-percept preferred FD was
computed analogously by averaging the FD of every image on which the
word was reported, restricted to words with at least 30 reports.
