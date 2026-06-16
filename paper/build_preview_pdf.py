"""Render an HTML preview of the manuscript and convert to PDF using
WeasyPrint. The output mirrors the .tex layout but compiles without LaTeX.

Style target: PNAS Research Article.
  * Significance Statement immediately after the abstract
  * Introduction → Results → Discussion → Materials and Methods (in order)
  * Declarative subsection titles ending with a period (PNAS house style)
  * No em or en dashes anywhere in the prose
"""
from __future__ import annotations

from pathlib import Path

from weasyprint import HTML, CSS

HERE = Path(__file__).parent
FIG = HERE / "figures"

CSS_STYLE = """
@page { size: A4; margin: 2.2cm 2.5cm;
         @bottom-center { content: counter(page); font-family: Arial;
                           font-size: 9pt; color: #666; } }
body  { font-family: Arial, sans-serif; font-size: 10.5pt;
         line-height: 1.5; color: #111; }
h1    { font-size: 17pt; margin-bottom: 0.2em; line-height: 1.25; }
h2    { font-size: 12.5pt; margin-top: 1.4em; margin-bottom: 0.3em;
         border-bottom: 1px solid #999; padding-bottom: 0.1em;
         text-transform: uppercase; letter-spacing: 0.04em; }
h3    { font-size: 11pt; margin-top: 1.0em; margin-bottom: 0.2em;
         font-weight: bold; }
h4    { font-size: 10.5pt; margin-top: 0.7em; margin-bottom: 0.1em;
         font-weight: bold; color: #222; }
/* PNAS-style declarative result heading: bold lead-in, runs with paragraph */
.lead { font-weight: bold; }
p     { text-align: justify; margin: 0 0 0.55em 0; }
em    { font-style: italic; }
.author { color: #555; margin-bottom: 0.4em; font-size: 10pt; }
.abstract, .significance {
  background: #f5f5f5; padding: 0.9em 1.2em;
  border-left: 3px solid #999; font-size: 10pt;
  line-height: 1.45; margin-bottom: 1.0em;
}
.significance { background: #f0f4ee; border-left-color: #5b7a4b; }
figure { margin: 1.0em 0 1.2em 0; text-align: center;
          page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 22cm; }
figure.narrow img { max-width: 60%; }
figure.half   img { max-width: 75%; }
figcaption { font-size: 9pt; color: #333; margin-top: 0.3em;
              text-align: justify; line-height: 1.4; }
figcaption strong { color: #000; }
code  { font-family: Consolas, monospace; font-size: 0.9em;
         background: #f0f0f0; padding: 0 3px; }
"""


def fig(src: str, caption_title: str, caption_body: str = "",
        label: str = "", css_class: str = "") -> str:
    cls = f' class="{css_class}"' if css_class else ""
    return (f'<figure id="{label}"{cls}>'
            f'<img src="figures/{src}" />'
            f'<figcaption><strong>{caption_title}</strong> '
            f'{caption_body}</figcaption></figure>')


HTML_BODY = f"""
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"></head><body>

<h1>Image statistics and trait creativity jointly shape visual pareidolia</h1>

<div class="author">[Authors] · Manuscript draft, PNAS format ·
generated from <code>paper/build_preview_pdf.py</code></div>

<div class="abstract">
<strong>Abstract.</strong>
Pareidolia, the perception of meaningful objects in ambiguous patterns,
has recently been linked to creativity as a marker of divergent
perception. Yet this relationship has not been examined at scale in a
way that disentangles the contribution of image properties from that
of the observer's creativity. Moreover, no study has systematically
examined the perceptual content of unconstrained pareidolia using
generated fractal images. Here, we collected 580 online sessions in
which participants viewed 30 fractal-noise patches spanning three
controlled fractal-dimension levels (FD ≈ 1.3, 1.5, 1.7) and freely
typed every percept that came to mind, before and after completing the
Divergent Association Task (DAT; Olson et al., 2021) as a stable
measure of trait creativity. A subset opted in to webcam-based eye
tracking. Across 18,500 unique percept reports we show that (i) image
complexity systematically reshapes both <em>what</em> people see,
shifting the modal percept from anthropomorphic figures at low
complexity to animals at intermediate complexity and to textural
descriptions at high complexity, and <em>how</em> they explore the
image; (ii) trait creativity predicts a small but reliable bias away
from human and toward animal percepts, higher semantic diversity at
intermediate complexity, and a measurable preference for specific
embodied or unusual categories. This study positions pareidolia as a
scalable paradigm for probing the perceptual dimensions of creative
cognition, suggesting that divergent thinking may be linked to
divergent perception through the semantic diversity of meanings that
observers extract from ambiguous visual forms.
</div>

<div class="significance">
<strong>Significance Statement.</strong>
Visual pareidolia, seeing faces and figures in clouds or rocks, is a
ubiquitous demonstration that perception is constructed jointly from
the image and from the observer. Existing studies typically test one
or the other in small samples. Here we vary the statistical complexity
of fractal-noise stimuli at three controlled levels and combine free
verbal report from approximately 580 online participants with a
canonical measure of trait creativity and optional webcam eye tracking.
The dual manipulation shows that image complexity controls the modal
percept on a humans-to-animals-to-texture continuum, while individual
creativity shifts <em>which</em> percepts are reported rather than
<em>how many</em>, refining what divergent-thinking traits contribute
to ambiguous visual perception.
</div>

<h2>Introduction</h2>

<p>Pareidolia, the perception of meaningful figures in unstructured
visual input, sits at the intersection of bottom-up image statistics
and top-down inference. Bayesian and predictive-coding accounts hold
that perception integrates noisy sensory evidence with internal priors
(Friston, 2010; Summerfield and de Lange, 2014; de Lange et al.,
2018), and pareidolia is a textbook case in which the prior dominates:
observers report faces in toast, animals in clouds, and figures in
rocks. Neuroimaging of face-pareidolia confirms that illusory faces
engage face-selective cortex, initially with the same temporal
signature as real faces before rapidly transitioning to that of
ordinary objects (Liu et al., 2014; Wardle et al., 2020); the
phenomenon is not unique to humans, with rhesus monkeys also showing
spontaneous face-pareidolia responses to object stimuli (Taubert et
al., 2017). Far less is known about the constructive side of the
process: <em>which</em> meanings the observer actually extracts when
the image affords many, and how that depends jointly on stimulus
statistics and observer disposition.</p>

<p>A converging line of evidence links the constructive side of
pareidolia to creativity. The classical associative account of creative
thinking (Mednick, 1962) holds that creative individuals have flatter
associative hierarchies, retrieving more remote concepts when prompted;
modern work has formalised this idea using semantic-network structure
and computational measures of semantic distance (Kenett et al., 2014;
Beaty and Johnson, 2021), and the Divergent Association Task (DAT)
provides a brief, language-based, test-retest reliable index of this
property by asking participants to name ten words that are as
different as possible (Olson et al., 2021). Bellemare-Pepin et al.
(2022) extended this principle from cognition to perception, showing
in a sample of 50 participants that high-creative observers
spontaneously report more, more diverse, and faster pareidolic
percepts from cloud-like fractal images than low-creative observers.
Bellemare-Pepin and Jerbi (2024) articulated this finding as
<em>divergent perception</em>, a perceptual counterpart to divergent
thinking in which flexibility in extracting meaning from ambiguous
sensory input is hypothesised to be a generalised cognitive trait.
Independently, individual differences in face pareidolia have been
linked to schizotypy, paranormal belief, and broader anomalous
perceptual experience (Riekki et al., 2013; Zhou and Meng, 2020;
Salge et al., 2021), and to the vividness of voluntary mental imagery
(Pearson, 2019; Salge et al., 2021), suggesting that pareidolia
indexes a stable readout dimension of the visual system.</p>

<p>Three obstacles have limited the scale at which these claims can be
tested. First, most adjacent pareidolia studies use binary
face/no-face decisions or low-trial paradigms with samples of dozens
(Liu et al., 2014; Riekki et al., 2013; Zhou and Meng, 2020;
Salge et al., 2021; Wardle et al., 2020), making it difficult to
disentangle image-side and observer-side contributions within the same
design. Second, the natural medium of pareidolia, free verbal report,
has historically been hard to score at scale; modern sentence
embeddings (Pennington et al., 2014; Reimers and Gurevych, 2019),
density-based clustering (Campello et al., 2013) and non-linear
dimensionality reduction (McInnes et al., 2018) make this tractable.
Third, the stimulus side has been dominated by curated face-like
images. Fractal-noise stimuli replace this curation with a single
controlled parameter, the fractal dimension (FD), which sets the slope
of the amplitude spectrum and the local structure of the resulting
image (Field, 1987; Knill et al., 1990; Spehar et al., 2016); humans
show systematic perceptual and aesthetic responses across the FD range,
peaking around FD ≈ 1.3 to 1.5 (Spehar et al., 2016). Browser-based
experimentation (de Leeuw, 2015) and webcam eye tracking (Papoutsaki
et al., 2016) then make it possible to assemble a controlled
image-by-observer factorial design at hundreds of participants.</p>

<p>Here we extend the Bellemare-Pepin (2022) paradigm to a much larger
scale and a richer dependent variable. Approximately 580 online
participants each viewed 30 fractal-noise patches drawn from a 300-image
bank at three calibrated fractal-dimension levels (FD ≈ 1.3, 1.5,
1.7; labelled FD12, FD14, FD16 by generation target), freely typed
every percept that came to mind, completed the DAT both before and
after the pareidolia task, and (when opted in) had their gaze
recorded with WebGazer. We ask four questions on the same participants.
First, does image complexity reshape the amount, the semantic content,
and the inter-observer consensus of pareidolia? Second, does trait
creativity shift the <em>kind</em> of percept reported beyond its
amount? Third, does completing the pareidolia task itself perturb DAT,
or is the link between DAT and pareidolia trait-driven? Fourth, does
successful pareidolia leave a measurable signature in eye movements?</p>

<h2>Results</h2>

<h3>Image complexity controls the amount of pareidolia.</h3>
<p>We first asked whether the image's statistical structure modulates
how readily participants extract a percept. As complexity rose from
FD12 to FD16, the average number of single words submitted per trial
decreased monotonically from 1.69 to 1.56 (Friedman
χ² = 46.5, <em>P</em> = 7.9 × 10<sup>-11</sup>; Fig. 1), and the
parallel decrease in the number of free-text descriptions per trial
was equally robust (<em>P</em> = 6.7 × 10<sup>-10</sup>). The
within-subject median semantic spread of percepts at each FD, a
participant-level index of how semantically scattered the answers were
on stimuli of a given complexity, also decreased slightly but
reliably (<em>P</em> = 0.02). Mean reaction time did not differ by FD
(<em>P</em> = 0.17). FD therefore acts as a behavioural manipulation
that biases observers toward fewer and progressively more uniform
percepts.</p>

{fig("fig1_fd_behaviour.png",
     "Fig. 1. Image complexity reduces the amount of pareidolia.",
     "Within-subject mean ± 95% CI by FD level for reaction time, "
     "single words per trial, descriptions per trial, and the "
     "within-subject median semantic spread of percepts. Brackets "
     "denote significant pair-wise Wilcoxon signed-rank contrasts.",
     "fig1")}

<h3>Image complexity reshapes the content of pareidolia.</h3>
<p>The drop in amount was accompanied by a striking shift in semantic
content. We assigned every unique percept word to one of fifteen
seed-based semantic buckets (face, body-part, person, mammal, bird,
fish-aquatic, insect-bug, reptile, creature, object, landscape,
weather, abstract, food, action) by nearest seed-centroid in BERT
space (Materials and Methods), with a hard-reject list preventing
high-frequency objects (e.g., <em>hat, cross, hammer, gun</em>) and
actions (e.g., <em>kiss, dancing</em>) from leaking into the human or
animal categories. Three highly significant within-subject FD effects
emerged on what participants named (Fig. 2). The proportion of
percepts falling in human-related categories (face, body-part, person)
dropped monotonically from 33% at FD12 to 25% at FD16
(χ² = 73.3, <em>P</em> = 1.3 × 10<sup>-15</sup>); the animal share
(mammal, bird, fish, insect, reptile) rose from 25% to a peak of 32%
at FD14 (<em>P</em> = 1.2 × 10<sup>-7</sup>); and the
animal-minus-human bias index inverted sign across the FD axis, from
clearly human dominated at FD12 (−0.16) to mildly animal dominated at
FD16 (+0.09; <em>P</em> = 6.8 × 10<sup>-13</sup>).</p>

<p>At the image rather than the participant level, the same complexity
gradient produced a parallel rise in inter-observer disagreement. The
Shannon entropy of the percept-word distribution per stimulus rose
from a median of 5.90 bits at FD12 to 6.05 bits at FD16
(Kruskal-Wallis <em>P</em> = 0.014; Fig. 2, rightmost panel),
indicating that high-complexity images do not converge on a canonical
answer the way low-complexity images do. The lexical shift is visible
in the most distinctive words per FD: FD12 percepts are dominated by
discrete, often anthropomorphic figures (<em>vase, handshake, elbow,
horn, pregnant, ape, cave</em>); FD14 by mammalian and small-animal
labels (<em>ox, warthog, bunnies, salamander, llama, mice,
rooster</em>); FD16 by texture and atmosphere descriptors
(<em>canopy, stars, fuzzy, marble, rain, scatter, noise, swamp</em>).
The progression from figural to animal to textural is also visible
spatially in a t-SNE projection of every unique percept word coloured
by majority FD level (Fig. 3b): FD12 words concentrate in face and
person regions, FD14 in the central animal cloud, and FD16 spreads
into the abstract and textural periphery.</p>

{fig("fig2_fd_perception.png",
     "Fig. 2. Image complexity reshapes the content of pareidolia.",
     "Within-subject mean ± 95% CI per FD of the human share, animal "
     "share, and animal-minus-human bias index (panels 1 to 3, bars), "
     "together with the per-image vocabulary entropy of percept "
     "distributions (rightmost panel, bar = mean across the ten images "
     "per FD). Brackets denote significant pair-wise contrasts "
     "(Wilcoxon for the participant-level metrics, Mann-Whitney U for "
     "the image-level panel).",
     "fig2")}
{fig("fig3_semantic_territory.png",
     "Fig. 3. Semantic territory of collective pareidolia.",
     "Shared t-SNE projection (cosine, perplexity = 30) of every "
     "unique percept word (n = 4,361); dot size scales with corpus "
     "frequency; KDE contours per group. (a) coloured by majority "
     "creativity tertile (sequential warm palette, amber = low, dark "
     "red = high DAT). (b) coloured by majority FD level at which the "
     "word was produced. Both panels share the same projection.",
     "fig3")}

<h3>Observers largely disagree on what they see, but agreement decreases
with complexity.</h3>
<p>Beyond the group-level shifts in modal percept, we asked how much
individual observers actually converge on the same answer for the same
image. For every stimulus (300 images, approximately 50 observers each)
we computed two complementary inter-observer agreement metrics: the
<em>modal-word share</em>, defined as the proportion of all percept
words on that image that equal the single most common word, and the
<em>mean cross-observer cosine similarity</em>, defined as the mean
BERT cosine similarity between two randomly sampled percept words
from <em>different</em> participants on the same image (the literal
same word yields 1.0; close synonyms yield ~0.6 to 0.8; unrelated
words yield ~0.1). Both metrics agree that consensus is low in
absolute terms: on average the single most common word accounts for
only 7% of all percepts (best image: 27%; worst image: 4%), and the
mean cross-observer cosine similarity sits around 0.29 (Fig. 4).
Pareidolia, at this resolution, is mostly an idiosyncratic act. The
small but reliable consensus core that does exist is modulated by
image complexity: low-FD images elicit higher agreement than high-FD
images on both metrics (Kruskal-Wallis on mean cosine
<em>P</em> = 0.004; on modal share <em>P</em> = 4 × 10<sup>-5</sup>),
with monotonic decreases across FD12 → FD14 → FD16. A complementary
per-(image, DAT-tertile) analysis paired across the 300 images found
no detectable effect of observer creativity on per-image consensus
(Friedman <em>P</em> = 0.70 on mean cosine, <em>P</em> = 0.22 on
vocabulary entropy), indicating that the ambiguity of an image is a
property of the image itself rather than of the creativity profile of
its observers.</p>

{fig("fig4_image_consensus.png",
     "Fig. 4. Inter-observer agreement decreases with image complexity.",
     "Per-image inter-observer agreement summarised across 300 stimuli "
     "(100 per FD level). (a) Mean cross-observer cosine similarity "
     "between two randomly sampled percept embeddings from different "
     "participants on the same image. (b) Modal-word share: fraction "
     "of all percept words on the image that equal the single most "
     "common word. Box-and-whisker plots show the distribution across "
     "the 100 images per FD level; dots are individual images. "
     "Brackets denote significant pair-wise Mann-Whitney U contrasts.",
     "fig4")}

<h3>Test-retest reliability of DAT is preserved across the task.</h3>
<p>Among the <em>n</em> = 507 participants who completed both DAT
timepoints (Fig. 5), pre- and post-task scores were positively but
moderately correlated (Pearson <em>r</em> = 0.19,
<em>P</em> = 1.2 × 10<sup>-5</sup>), with no detectable group-level
shift (Wilcoxon signed-rank <em>W</em> = 6.1 × 10<sup>4</sup>,
<em>P</em> = 0.99; mean Δ = −0.02, SD = 4.4). The link between DAT
and pareidolia reported below replicates consistently with the
pre-task score and weakens substantially when the post-task score is
substituted in, consistent with this link capturing a stable
trait-level property of the observer rather than a transient state
shift induced by the pareidolia task.</p>

{fig("fig5a_dat_pre_post.png",
     "Fig. 5. Pre vs post DAT scores.",
     "Left: GloVe-scored DAT post vs DAT pre, n = 507. Right: "
     "distribution of the participant-level delta (post minus pre).",
     "fig5a", css_class="narrow")}

<h3>Trait creativity does not predict more or faster pareidolia.</h3>
<p>Across six general behavioural metrics (fraction of trials
completed, single words per trial, descriptions per trial, mean
reaction time, total unique words across the experiment, and mean
corpus rarity of a participant's words), only one survived bivariate
±3 SD outlier trimming as significantly related to DAT, and in the
opposite direction to a "more is more" account: high-creativity
participants completed slightly <em>fewer</em> trials than
low-creativity participants (<em>r</em> = −0.10, <em>P</em> = 0.03;
Fig. 6). Words per trial and reaction time were both null.
Creativity, in this dataset, therefore does not manifest as producing
more or faster pareidolia.</p>

{fig("fig5b_dat_behaviour.png",
     "Fig. 6. Creativity is not associated with more or faster "
     "pareidolia.",
     "DAT vs mean words per trial (left) and DAT vs mean reaction "
     "time (right). The number of percepts extracted per trial and "
     "the time taken to extract them do not covary with creativity.",
     "fig5b", css_class="narrow")}

<h3>Creativity predicts greater perceptual diversity, concentrated at
intermediate complexity.</h3>
<p>The within-subject median pair-wise cosine distance of a
participant's percept embeddings, a participant-level measure of how
semantically scattered the percepts are, was positively correlated
with DAT (<em>r</em> = +0.14, <em>P</em> = 0.001; <em>n</em> = 500;
Fig. 7, left), reproducing a previously reported relationship between
divergent thinking and lexical diversity in a perceptual task.
Splitting the analysis by FD revealed that the effect concentrates at
the middle complexity level (Fig. 7, right): <em>r</em> = +0.17
(<em>P</em> = 2.7 × 10<sup>-4</sup>) at FD14, <em>r</em> = +0.09
(<em>P</em> = 0.04) at FD12, and essentially null at FD16
(<em>r</em> ≈ 0, <em>P</em> = 0.93). Intermediate complexity therefore
appears to be the regime in which divergent-thinking ability has the
greatest expressive room.</p>

{fig("fig5cd_diversity.png",
     "Fig. 7. Creativity predicts perceptual diversity, especially at "
     "intermediate complexity.",
     "(left) pooled across all FD levels (one point per participant). "
     "(right) the same analysis split by FD: the relationship "
     "concentrates at FD14, the middle complexity level.",
     "fig5cd")}

<h3>Creativity biases pareidolia from humans toward animals.</h3>
<p>The same fifteen-bucket taxonomy pinpointed where the link between
DAT and diversity comes from. The per-participant share of percepts
falling in human categories (face + body-part + person) was negatively
associated with DAT (<em>r</em> = −0.14, <em>P</em> = 0.003), while
the animal share was numerically higher in high-creativity
participants but did not reach significance on its own
(<em>r</em> = +0.08, <em>P</em> = 0.10; Fig. 8). Combining the two
into an animal-minus-human bias index produced the cleanest
creativity effect we observed (<em>r</em> = +0.12, <em>P</em> = 0.006;
Spearman ρ = +0.14, <em>P</em> = 0.001). High-creativity participants
therefore do not extract <em>more</em> pareidolic percepts overall.
Rather, they shift the <em>kind</em> of percept they extract, away
from canonical human and face responses and toward animal ones.</p>

{fig("fig5f_dat_human_animal_bias.png",
     "Fig. 8. Creativity tilts the percept distribution away from "
     "humans and toward animals.", "", "fig5f")}

<h3>Unsupervised clusters separate low- and high-creativity profiles.</h3>
<p>The human-vs-animal contrast captures one axis along which
creativity restructures percepts, but the seed-based taxonomy is
deliberately narrow. To examine the full preference landscape we
re-derived categories <em>de novo</em> by clustering all percept words
that occurred at least three times in the corpus (Materials and
Methods). HDBSCAN yielded 32 interpretable clusters plus a small
noise group (22% of words); about a third of clusters had a 95%
bootstrap CI on the log-odds (high vs low tertile) preference that
excluded zero (Fig. 9b).</p>

<p>High-creativity participants were over-represented in clusters of
<em>specific, embodied, or unusual</em> percepts:
<em>mushroom / apple / clover / carrot</em>;
<em>gremlin / ogre / gargoyle / ghoul</em>;
<em>blood / stomach / kidney / vomit</em>;
<em>fish / seahorse / crab / whale</em>;
<em>bat / vase / ball / cob</em>;
<em>hand / finger / fist / hands</em>;
<em>witch / cartoon / demon / angel</em>. Low-creativity participants
were over-represented in clusters of <em>abstract or texture-like</em>
percepts: <em>plus / separation / two / open</em>;
<em>water / spill / mess / pipe</em>;
<em>ink / paint / painting / book</em>;
<em>ocean / lake / river / sea</em>;
<em>land / mountain / cliff</em>;
<em>fire / explosion / death</em>;
<em>hammer / hole / splatter / dots</em>. The semantic-map view
(Fig. 9a) makes the structure of the contrast explicit: the two
preference profiles occupy non-overlapping regions of BERT space,
with the creature, viscera, and specific-object region on the upper
side and the landscape, texture, and abstract region on the lower
side. Together with the bias-index result above, this argues that
the creativity effect on pareidolia is best characterised not as
seeing <em>more</em> but as seeing <em>different</em> things.</p>

{fig("fig6_preferred_categories.png",
     "Fig. 9. Percept clusters preferred by low vs high creativity.",
     "(a) 2D UMAP projection of every percept word with at least 3 "
     "corpus occurrences, coloured by the log-odds preference of its "
     "HDBSCAN cluster (red = preferred by high-creativity "
     "participants, amber = preferred by low-creativity). Labels mark "
     "the top six clusters on each side at their UMAP centroid. (b) "
     "Forest plot of all 32 retained clusters, log-odds (high/low) "
     "with bootstrap 95% CI; stars mark clusters whose CI excludes "
     "zero. Numbers on the right are word occurrences per cluster.",
     "fig6")}

<h3>The same pipeline reveals a complementary FD preference axis.</h3>
<p>Repurposing the clustering with stimulus FD level as the contrast
variable (FD16 vs FD12 instead of high vs low DAT) shows partly
overlapping but distinct preference geometries (Fig. 10). High-FD
images are over-represented in clusters of <em>tree / forest</em>,
<em>butterfly / caterpillar / worm</em>, <em>alien / sky / space</em>,
<em>fish / seahorse / crab</em>, <em>witch / demon / angel</em>, and
<em>ghost / lamp / chaos</em>, whereas low-FD images dominate
clusters of <em>cave / rock / stones</em>,
<em>blood / stomach / kidney</em>, <em>boot / foot / arm</em>,
<em>bat / vase / ball</em>, and <em>land / mountain / cliff</em>.
Creativity drives a humans-to-animals shift along one axis of the BERT
space; image complexity drives a body-parts-to-atmospheric-creatures
shift along another. The two manipulations therefore probe partly
independent dimensions of the same percept landscape.</p>

{fig("fig7_preferred_categories_fd.png",
     "Fig. 10. Percept clusters preferred by low vs high stimulus FD.",
     "Same clustering pipeline as Fig. 9; the contrast variable is "
     "FD16 versus FD12. (a) Semantic map coloured by log-odds "
     "preference (blue = FD12, red = FD16). (b) Forest plot of "
     "clusters, log-odds (FD16/FD12) with bootstrap 95% CI; stars "
     "mark clusters whose CI excludes zero.",
     "fig7")}

<h3>Gaze signatures of successful pareidolia.</h3>
<p>For the 247 participants who opted in to webcam-based gaze
recording and produced trials that survived the WebGazer
tracking-failure check (Materials and Methods), we computed six gaze
metrics per trial spanning fixation structure (count, mean duration),
scanpath length, and three spatial-concentration measures (gaze
entropy in bits, Gini coefficient of the 2D gaze histogram, and
recurrence rate of nearby fixations). We contrasted these metrics
along three independent axes: trials on which the participant did vs
did not report a pareidolic percept, stimulus FD level, and continuous
DAT score (Fig. 11).</p>

<p>Within the 105 participants who had at least two trials of each
kind, trials in which the participant submitted at least one word
("pareidolia") differed from no-word trials on three of the six
metrics, all in the same direction: more fixations per trial
(9.0 vs 9.4, Wilcoxon <em>P</em> = 0.045); lower 2D gaze entropy
(3.76 vs 3.71 bits, <em>P</em> = 0.037); and a higher Gini
concentration of the gaze distribution (0.868 vs 0.872,
<em>P</em> = 0.047). Fixation duration, scanpath length, and
recurrence rate did not differ. Successful pareidolia is therefore
accompanied by a small but reliable shift toward <em>more</em>
fixations clustered in a <em>smaller</em> region, consistent with the
participant having found something specific in the noise that they
are inspecting.</p>

<p>Within-subject FD effects emerged on scanpath length
(<em>P</em> = 0.006; monotonic decrease from FD12 to FD16) and,
marginally, on fixation duration (<em>P</em> = 0.06; longer fixations
at FD16). The other four metrics were not modulated by FD. As image
complexity rises, participants therefore move their eyes less and
dwell on a few central locations for longer, consistent with the
behavioural and content-level FD effects in Figs. 1 and 2. None of
the six metrics correlated reliably with DAT score (|<em>r</em>| &lt;
0.07 for all six; all <em>P</em> &gt; 0.30). Together with the
behavioural-engagement null reported above, this argues that
creativity does not change <em>how</em> people look at the stimulus,
only <em>what</em> they extract from it.</p>

{fig("fig4_eye_tracking.png",
     "Fig. 11. Eye-tracking metrics across three contrasts.",
     "Rows: (A) pareidolia vs no-pareidolia trials (within-subject "
     "Wilcoxon, n = 105 participants with at least 2 trials of each "
     "kind); (B) stimulus FD level (within-subject Friedman plus "
     "pair-wise Wilcoxon, n = 226); (C) continuous DAT score "
     "(between-subject Pearson, n = 198 to 227 depending on the "
     "metric, ±3 SD bivariate trim). Columns: six gaze metrics "
     "mixing classical (fixation count, fixation duration, scanpath "
     "length, gaze entropy) and recent measures (Gini concentration "
     "of the 2D gaze histogram, RQA-style recurrence rate of nearby "
     "fixations). Brackets mark significant pair-wise contrasts; "
     "small in-axis text on row C reports the Pearson r.",
     "fig4")}

<h2>Discussion</h2>

<p>Three findings emerge consistently across analyses. First, the
statistical complexity of the noise determines the modal pareidolic
percept on a human-to-animal-to-texture continuum, with the
between-image variability of percepts growing as complexity rises.
Second, trait creativity, as indexed by the DAT, does not modify how
much pareidolia an observer produces but shifts which categories the
observer favours, away from canonical human and face percepts and
toward animals, viscera, and specific embodied objects. Third, both
manipulations leave converging gaze signatures: successful pareidolia
is accompanied by more concentrated fixations, and rising image
complexity tightens the scan pattern, while creativity has no
detectable gaze signature.</p>

<p>The asymmetry between the rich content effects of creativity and
its null behavioural and oculomotor effects is informative. It
suggests that what divergent-thinking measures track in this paradigm
is not a bias toward sensitivity (more percepts, faster percepts) but
a bias toward unusual readout of an otherwise comparable extraction
process. A practical consequence is that pareidolia tasks designed to
maximise individual differences should score the kind of percept
rather than its presence or its count. A second consequence is that
mid-complexity stimuli (FD14 in our parameterisation) are the regime
in which observer-driven variability is most expressible.</p>

<p>The pre versus post DAT result, near-zero mean shift but moderate
test-retest correlation, supports the interpretation of DAT as a
stable trait rather than a state, and suggests that completing
approximately 15 minutes of pareidolia does not measurably alter
divergent-thinking ability. The strong attenuation of links between
post-task DAT and pareidolia, alongside the preserved links to
pre-task DAT, further supports a trait reading.</p>

<p>Several limitations should be noted. Webcam eye tracking, although
adequate for the coarse-grained contrasts reported here, lacks the
precision to resolve fixation-level effects within the image; the
moderate-strength gaze effects we report are unlikely to be
overestimates. The semantic categorisation rests on contemporary
sentence embeddings and inherits whatever biases those embeddings
carry; we mitigated this by combining a seed-based taxonomy with an
unsupervised HDBSCAN view, and the two views agreed on the principal
contrasts. Finally, the design samples three discrete FD levels rather
than a continuum; a finer sweep would test the monotonicity claims
more strictly.</p>

<h2>Materials and Methods</h2>

<h3>Participants.</h3>
<p>Participants were recruited online via [recruitment platform]; all
gave informed consent in accordance with the [ethics committee]
guidelines. The full session was completed by <em>N</em> = 579
English-speaking adults (the "main cohort"), comprising a demographic
form, the 10-word Divergent Association Task (Olson et al., 2021) as
a baseline measure of trait creativity (DAT pre), an optional 9-point
WebGazer calibration and validation routine, 30 pareidolia trials with
free-response word entry, a repeated DAT (DAT post; <em>n</em> = 507
provided a scoreable response), and a closing feedback questionnaire.
The main cohort analysed below is defined as the intersection of (i)
English as primary language, (ii) feedback supplied, and (iii) at
least one valid pareidolia trial. Sessions whose stored DAT word list
duplicated one of the previous two log entries were removed.</p>

<h3>Stimuli.</h3>
<p>The stimulus set comprised 30 greyscale fractal-noise images, ten
at each of three fractal-dimension levels labelled <strong>FD12, FD14,
FD16</strong> by their generation target. Each image was synthesised
by filtering 2D Gaussian noise so that its amplitude spectrum
followed a 1/<em>f</em><sup>β</sup> profile prior to inverse Fourier
transform; higher FD produces finer-grained spatial detail and lower
local autocorrelation. The realised fractal dimension of every
generated image was estimated empirically with the standard 2D
box-counting method (Liebovitch and Toth, 1989); across the 30 images
the mean ± SD per nominal level was FD12: 1.27 ± 0.012,
FD14: 1.49 ± 0.016, FD16: 1.67 ± 0.011 (100 images per level were
generated and the 10 closest to each target were retained). For
readability we refer to the three conditions throughout the
manuscript as FD ≈ 1.3, 1.5, and 1.7. Each participant saw all 30
images in a fully randomised order, each rendered centred on a white
viewport with <code>max-width: 55%</code>.</p>

<h3>Pareidolia task.</h3>
<p>A pareidolia trial began with a 500 ms central fixation cross,
followed by the stimulus image presented for up to 30 s, terminated
early if the participant pressed the spacebar to advance. Immediately
after stimulus offset a free-text response screen appeared with five
short-answer slots labelled "Single words" and five longer slots
labelled "Descriptions (optional)"; participants were instructed to
write every distinct percept that the image evoked. When eye tracking
had been enabled at session start, raw (<em>x, y, t</em>) gaze samples
were captured throughout image presentation via the WebGazer
browser-side extension to jsPsych.</p>

<h3>Percept word preprocessing.</h3>
<p>Each typed entry was lowercased and whitespace-stripped. To
minimise contamination by non-percept entries we kept only entries
matching the regular expression <code>^[A-Za-z]+$</code> (single
alphabetic tokens without spaces, digits, or punctuation) and
additionally removed the catch-all stopwords <em>nothing,
nothingness, black, white, map, cloud, clouds</em>. Sentence-level
descriptions were excluded from semantic analyses to avoid mixing
single-word and multi-word vector representations, which occupy
systematically different regions of embedding space. All semantic
computations use 384-dimensional sentence-transformer embeddings
(<code>all-MiniLM-L6-v2</code>) of every unique surviving percept
word (<em>n</em><sub>words</sub> = 4,361).</p>

<h3>Divergent Association Task (DAT) scoring.</h3>
<p>Both DAT pre and DAT post were scored with the canonical Olson et
al. (2021) procedure. Each set of 10 candidate words was lowercased
and restricted to alphabetic forms; the first seven words with a
GloVe 840B 300-d vector were retained, and the DAT score was defined
as 100 × mean(<em>d<sub>ij</sub></em>), where <em>d<sub>ij</sub></em>
is the cosine distance between every pair of those seven vectors.
Sessions producing fewer than seven valid lookups yielded no DAT
score and were excluded from any analysis using that timepoint. DAT
scores in main-text analyses refer to the baseline (pre) score unless
explicitly noted.</p>

<h3>Eye-tracking preprocessing.</h3>
<p>The online WebGazer calibration sweep provides a dense set of gaze
samples spanning most of the viewport. For each session, we estimated
the usable gaze extent as the 1st-to-99th-percentile box of these
calibration samples and used it to rescale all subsequent gaze data
into the unit square [0, 1], eliminating cross-session differences in
screen size and browser viewport. The stimulus region of interest
(ROI) was approximated as a per-session rectangle centred on the
recorded stimulus position with half-width 0.275 (half of the
<code>55%</code> CSS rule) and half-height 0.275 · <em>W/H</em>, where
<em>W/H</em> is the session's viewport aspect ratio. Trials whose
normalised gaze trace formed a near-perfect line, defined as
|corr(<em>x, y</em>)| &gt; 0.97 together with span &gt; 0.6 on both
axes, a signature of the WebGazer face-mesh tracker failing onto a
screen edge, were flagged as tracking failures and dropped (530 of
5,660 trials, 9.4%). Fixations were detected with a
dispersion-threshold (I-DT) algorithm using a dispersion bound of
0.05 in normalised coordinates and a minimum duration of 100 ms.</p>

<h3>Per-participant and per-image semantic metrics.</h3>
<p>For every participant and every FD level we computed (i) the
<em>semantic spread</em> of percepts as the median pair-wise cosine
distance over their unique word embeddings, and (ii) the
<em>vocabulary surprisal</em> as the mean −log <em>p</em> over the
corpus frequency of each word. To examine the <em>kind</em> of percept
rather than its amount, we assigned every unique word to one of
fifteen semantic categories (<em>face, body-part, person, mammal,
bird, fish-aquatic, insect-bug, reptile, creature, object, landscape,
weather, abstract, food, action</em>) by nearest seed-centroid in BERT
space (minimum cosine ≥ 0.30, otherwise "other"). Seed lists were
hand-curated and audited against the highest-frequency words in the
corpus; an explicit hard-reject list prevents a small number of
items that are close to a human or animal centroid in embedding
space but are clearly not (objects: <em>hat, cross, hammer, gun, ball,
bat, vase</em>; actions: <em>kiss, kissing, dancing, dance, hugging</em>;
landscape: <em>sea, lake, ocean, river</em>) from being assigned to
those categories; such items are routed to their best non
human/animal centroid instead. For each (participant, FD) we
summarised the percept distribution as the <em>human share</em>
(face + body-part + person), the <em>animal share</em>
(mammal + bird + fish + insect + reptile), and the <em>animal-minus-human
bias index</em> (<em>A − H</em>) / (<em>A + H</em>), ranging from −1
(entirely human) to +1 (entirely animal). At the image level we
computed the Shannon entropy of the percept-word distribution
elicited by each stimulus, as a measure of inter-observer consensus.</p>

<h3>Unsupervised clustering of percepts.</h3>
<p>To complement the seed-based taxonomy with an unsupervised view,
we clustered every percept word that appeared at least three times in
the corpus (<em>n</em> = 1,110 words). The BERT embeddings were L2
normalised, reduced to 10 dimensions with UMAP (cosine metric,
<code>n_neighbors</code> = 15, <code>min_dist</code> = 0), and
clustered with HDBSCAN (<code>min_cluster_size</code> = 12,
<code>cluster_selection_method</code> = EOM). The procedure produced
32 interpretable clusters plus a noise group (22% of words). Each
cluster was labelled by its top four log-odds most distinctive words,
i.e., the words whose probability inside the cluster most exceeds
their probability outside it.</p>

<h3>Statistical analyses.</h3>
<p>Within-subject effects of FD were tested with Friedman's χ²
followed by pair-wise Wilcoxon signed-rank contrasts; only contrasts
significant at <em>P</em> &lt; 0.05 are annotated on the figures
(brackets with asterisks). Between-subject DAT effects were tested
with Pearson's <em>r</em> (reported in the main figures) and
Spearman's ρ (reported in the companion CSV tables). Before every
correlation we trimmed bivariate outliers at ±3 SD on both the
dependent metric and the DAT score; no other transformations were
applied. For the cluster-preference forest plot, 95% bootstrap CIs
(2,000 binomial resamples per cluster) were computed on the log-odds
of the (high/low) tertile share. Statistics are reported as <em>r</em>
with significance markers
(<sup>*</sup><em>P</em> &lt; 0.05,
<sup>**</sup><em>P</em> &lt; 0.01,
<sup>***</sup><em>P</em> &lt; 0.001); exact <em>P</em> values and
Spearman counterparts are given in the supplementary data tables.
Analyses were performed in Python 3.10 using <code>pandas</code>,
<code>numpy</code>, <code>scipy.stats</code>,
<code>sentence-transformers</code>, <code>umap-learn</code>, and
<code>hdbscan</code>; reproducible scripts and CSV outputs are
available at [URL].</p>

</body></html>
"""


def main():
    out_html = HERE / "draft_preview.html"
    out_pdf  = HERE / "draft_preview.pdf"
    out_html.write_text(HTML_BODY, encoding="utf-8")
    HTML(string=HTML_BODY, base_url=str(HERE)).write_pdf(
        target=str(out_pdf), stylesheets=[CSS(string=CSS_STYLE)],
    )
    print(f"Wrote {out_html.name} and {out_pdf.name}")


if __name__ == "__main__":
    main()
