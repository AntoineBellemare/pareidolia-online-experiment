/* Build the PNAS-style manuscript as a Word .docx using docx-js.
 *
 * Run:
 *   node paper/build_word.js
 *
 * Output:
 *   paper/draft.docx
 *
 * Content mirrors paper/build_preview_pdf.py. Figures are embedded as
 * PNGs from paper/figures/.
 */
const fs = require("fs");
const path = require("path");
const sharp = (() => { try { return require("sharp"); } catch (_) { return null; } })();
const imageSizeMod = require("image-size");
const sizeOf = imageSizeMod.imageSize || imageSizeMod.default || imageSizeMod;

const {
  Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType,
  HeadingLevel, ShadingType, BorderStyle, PageBreak, LevelFormat,
  TabStopType, TabStopPosition,
} = require("docx");

const PAPER_DIR = path.join(__dirname);
const FIG_DIR   = path.join(PAPER_DIR, "figures");
const OUT_PATH  = path.join(PAPER_DIR, "draft.docx");

// ── helpers ──────────────────────────────────────────────────────────────────

// US Letter, 1-inch margins -> 6.5" content. At 96 dpi display, 6.5 in = 624 px.
const CONTENT_W_PX = 624;

function imgDims(p) {
  const buf = fs.readFileSync(p);
  const { width, height } = sizeOf(buf);
  return { width, height, buf };
}

/**
 * Build a centred figure paragraph with caption.
 *   width: desired display width in px (default 580). For NARROW figures
 *          (single-column callers like 5a/5b), pass ~360.
 */
function fig(filename, captionStrong, captionBody, width = 580) {
  const fp = path.join(FIG_DIR, filename);
  const { width: w, height: h, buf } = imgDims(fp);
  const displayW = Math.min(width, CONTENT_W_PX);
  const displayH = Math.round(displayW * (h / w));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 80 },
      children: [
        new ImageRun({
          type: "png",
          data: buf,
          transformation: { width: displayW, height: displayH },
          altText: { title: captionStrong, description: captionBody,
                     name: filename },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 200 },
      children: [
        new TextRun({ text: captionStrong + " ", bold: true, size: 19 }),
        new TextRun({ text: captionBody, size: 19 }),
      ],
    }),
  ];
}

function P(text, opts = {}) {
  const { italic = false, bold = false, justify = true, after = 120 } = opts;
  return new Paragraph({
    alignment: justify ? AlignmentType.JUSTIFIED : AlignmentType.LEFT,
    spacing: { after, line: 320 },
    children: [new TextRun({ text, italic, bold, size: 22 })],
  });
}

/**
 * Rich paragraph builder. `runs` is an array of objects:
 *   { t: "text", bold?: true, italic?: true, sup?: true, sub?: true }
 */
function R(runs, { justify = true, after = 120 } = {}) {
  return new Paragraph({
    alignment: justify ? AlignmentType.JUSTIFIED : AlignmentType.LEFT,
    spacing: { after, line: 320 },
    children: runs.map(r => new TextRun({
      text: r.t,
      bold: !!r.bold,
      italics: !!r.italic,
      superScript: !!r.sup,
      subScript: !!r.sub,
      font: r.code ? "Consolas" : undefined,
      size: 22,
    })),
  });
}

function H1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 0, after: 120 },
    children: [new TextRun({ text, bold: true, size: 36 })],
  });
}
function H2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 320, after: 140 },
    children: [new TextRun({ text: text.toUpperCase(), bold: true,
                             size: 24, color: "333333" })],
  });
}
function H3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 220, after: 80 },
    children: [new TextRun({ text, bold: true, size: 22 })],
  });
}

function boxed(label, body, bg = "F4F4F4") {
  // Use a single-cell table to draw a tinted box around the abstract /
  // significance statement. Word renders this as a coloured panel.
  const { Table, TableRow, TableCell, WidthType } = require("docx");
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({
      children: [new TableCell({
        width: { size: 9360, type: WidthType.DXA },
        shading: { fill: bg, type: ShadingType.CLEAR },
        margins: { top: 200, bottom: 200, left: 240, right: 240 },
        borders: {
          left:  { style: BorderStyle.SINGLE, size: 18, color: "888888" },
          top:    { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
          right:  { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
          bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
        },
        children: [
          new Paragraph({
            spacing: { after: 120, line: 300 },
            children: [
              new TextRun({ text: label + " ", bold: true, size: 21 }),
              ...body.map(part => new TextRun({
                text: part.t, italics: !!part.italic, size: 21,
              })),
            ],
          }),
        ],
      })],
    })],
  });
}

// ── content ──────────────────────────────────────────────────────────────────

const children = [];

children.push(new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { after: 80 },
  children: [new TextRun({
    text: "Image statistics and trait creativity jointly shape visual pareidolia",
    bold: true, size: 34,
  })],
}));
children.push(new Paragraph({
  spacing: { after: 240 },
  children: [new TextRun({
    text: "[Authors] · Manuscript draft, PNAS format",
    italics: true, size: 20, color: "555555",
  })],
}));

children.push(boxed("Abstract.", [
  { t: "Pareidolia, the perception of meaningful objects in ambiguous " +
        "patterns, has recently been linked to creativity as a marker of " +
        "divergent perception. Yet this relationship has not been examined " +
        "at scale in a way that disentangles the contribution of image " +
        "properties from that of the observer’s creativity. Moreover, no " +
        "study has systematically examined the perceptual content of " +
        "unconstrained pareidolia using generated fractal images. Here, we " +
        "collected 580 online sessions in which participants viewed 30 " +
        "fractal-noise patches spanning three controlled fractal-dimension " +
        "levels (FD ≈ 1.3, 1.5, 1.7) and freely typed every percept that " +
        "came to mind, before and after completing the Divergent " +
        "Association Task (DAT; Olson et al., 2021) as a stable measure of " +
        "trait creativity. A subset opted in to webcam-based eye tracking. " +
        "Across 18,500 unique percept reports we show that (i) image " +
        "complexity systematically reshapes both " },
  { t: "what", italic: true },
  { t: " people see, shifting the modal percept from anthropomorphic " +
        "figures at low complexity to animals at intermediate complexity " +
        "and to textural descriptions at high complexity, and " },
  { t: "how", italic: true },
  { t: " they explore the image; (ii) trait creativity predicts a small " +
        "but reliable bias away from human and toward animal percepts, " +
        "higher semantic diversity at intermediate complexity, and a " +
        "measurable preference for specific embodied or unusual " +
        "categories. This study positions pareidolia as a scalable " +
        "paradigm for probing the perceptual dimensions of creative " +
        "cognition, suggesting that divergent thinking may be linked to " +
        "divergent perception through the semantic diversity of meanings " +
        "that observers extract from ambiguous visual forms." },
]));
children.push(new Paragraph({ children: [new TextRun({ text: "", size: 14 })] }));

children.push(boxed("Significance Statement.", [
  { t: "Visual pareidolia, seeing faces and figures in clouds or rocks, is a " +
        "ubiquitous demonstration that perception is constructed jointly " +
        "from the image and from the observer. Existing studies typically " +
        "test one or the other in small samples. Here we vary the " +
        "statistical complexity of fractal-noise stimuli at three controlled " +
        "levels and combine free verbal report from approximately 580 online " +
        "participants with a canonical measure of trait creativity and " +
        "optional webcam eye tracking. The dual manipulation shows that " +
        "image complexity controls the modal percept on a " +
        "humans-to-animals-to-texture continuum, while individual creativity " +
        "shifts " },
  { t: "which", italic: true },
  { t: " percepts are reported rather than " },
  { t: "how many", italic: true },
  { t: ", refining what divergent-thinking traits contribute to ambiguous " +
        "visual perception." },
], "EDF1E8"));

// ── INTRODUCTION ────────────────────────────────────────────────────────────
children.push(H2("Introduction"));

children.push(P(
  "Pareidolia, the perception of meaningful figures in unstructured " +
  "visual input, sits at the intersection of bottom-up image " +
  "statistics and top-down inference. Bayesian and predictive-coding " +
  "accounts hold that perception integrates noisy sensory evidence " +
  "with internal priors (Friston, 2010; Summerfield and de Lange, " +
  "2014; de Lange et al., 2018), and pareidolia is a textbook case " +
  "in which the prior dominates: observers report faces in toast, " +
  "animals in clouds, and figures in rocks. Neuroimaging of " +
  "face-pareidolia confirms that illusory faces engage face-selective " +
  "cortex, initially with the same temporal signature as real faces " +
  "before rapidly transitioning to that of ordinary objects (Liu et " +
  "al., 2014; Wardle et al., 2020); the phenomenon is not unique to " +
  "humans, with rhesus monkeys also showing spontaneous " +
  "face-pareidolia responses to object stimuli (Taubert et al., " +
  "2017). Far less is known about the constructive side of the " +
  "process: which meanings the observer actually extracts when the " +
  "image affords many, and how that depends jointly on stimulus " +
  "statistics and observer disposition."
));
children.push(R([
  { t: "A converging line of evidence links the constructive side of " +
        "pareidolia to creativity. The classical associative account of " +
        "creative thinking (Mednick, 1962) holds that creative individuals " +
        "have flatter associative hierarchies, retrieving more remote " +
        "concepts when prompted; modern work has formalised this idea " +
        "using semantic-network structure and computational measures of " +
        "semantic distance (Kenett et al., 2014; Beaty and Johnson, " +
        "2021), and the Divergent Association Task (DAT) provides a " +
        "brief, language-based, test-retest reliable index of this " +
        "property by asking participants to name ten words that are as " +
        "different as possible (Olson et al., 2021). Bellemare-Pepin et " +
        "al. (2022) extended this principle from cognition to perception, " +
        "showing in a sample of 50 participants that high-creative " +
        "observers spontaneously report more, more diverse, and faster " +
        "pareidolic percepts from cloud-like fractal images than " +
        "low-creative observers. Bellemare-Pepin and Jerbi (2024) " +
        "articulated this finding as " },
  { t: "divergent perception", italic: true },
  { t: ", a perceptual counterpart to divergent thinking in which " +
        "flexibility in extracting meaning from ambiguous sensory input " +
        "is hypothesised to be a generalised cognitive trait. " +
        "Independently, individual differences in face pareidolia have " +
        "been linked to schizotypy, paranormal belief, and broader " +
        "anomalous perceptual experience (Riekki et al., 2013; Zhou and " +
        "Meng, 2020; Salge et al., 2021), and to the vividness of " +
        "voluntary mental imagery (Pearson, 2019; Salge et al., 2021), " +
        "suggesting that pareidolia indexes a stable readout dimension " +
        "of the visual system." },
]));
children.push(P(
  "Three obstacles have limited the scale at which these claims can " +
  "be tested. First, most adjacent pareidolia studies use binary " +
  "face/no-face decisions or low-trial paradigms with samples of " +
  "dozens (Liu et al., 2014; Riekki et al., 2013; Zhou and Meng, " +
  "2020; Salge et al., 2021; Wardle et al., 2020), making it " +
  "difficult to disentangle image-side and observer-side " +
  "contributions within the same design. Second, the natural medium " +
  "of pareidolia, free verbal report, has historically been hard to " +
  "score at scale; modern sentence embeddings (Pennington et al., " +
  "2014; Reimers and Gurevych, 2019), density-based clustering " +
  "(Campello et al., 2013) and non-linear dimensionality reduction " +
  "(McInnes et al., 2018) make this tractable. Third, the stimulus " +
  "side has been dominated by curated face-like images. " +
  "Fractal-noise stimuli replace this curation with a single " +
  "controlled parameter, the fractal dimension (FD), which sets the " +
  "slope of the amplitude spectrum and the local structure of the " +
  "resulting image (Field, 1987; Knill et al., 1990; Spehar et al., " +
  "2016); humans show systematic perceptual and aesthetic responses " +
  "across the FD range, peaking around FD ≈ 1.3 to 1.5 (Spehar et " +
  "al., 2016). Browser-based experimentation (de Leeuw, 2015) and " +
  "webcam eye tracking (Papoutsaki et al., 2016) then make it " +
  "possible to assemble a controlled image-by-observer factorial " +
  "design at hundreds of participants."
));
children.push(R([
  { t: "Here we extend the Bellemare-Pepin (2022) paradigm to a much " +
        "larger scale and a richer dependent variable. Approximately 580 " +
        "online participants each viewed 30 fractal-noise patches drawn " +
        "from a 300-image bank at three calibrated fractal-dimension " +
        "levels (FD ≈ 1.3, 1.5, 1.7; labelled FD12, FD14, FD16 by " +
        "generation target), freely typed every percept that came to " +
        "mind, completed the DAT both before and after the pareidolia " +
        "task, and (when opted in) had their gaze recorded with " +
        "WebGazer. We ask four questions on the same participants. First, " +
        "does image complexity reshape the amount, the semantic content, " +
        "and the inter-observer consensus of pareidolia? Second, does " +
        "trait creativity shift the " },
  { t: "kind", italic: true },
  { t: " of percept reported beyond its amount? Third, does completing " +
        "the pareidolia task itself perturb DAT, or is the link between " +
        "DAT and pareidolia trait-driven? Fourth, does successful " +
        "pareidolia leave a measurable signature in eye movements?" },
]));

// ── RESULTS ─────────────────────────────────────────────────────────────────
children.push(H2("Results"));

children.push(H3("Image complexity controls the amount of pareidolia."));
children.push(R([
  { t: "We first asked whether the image’s statistical structure " +
        "modulates how readily participants extract a percept. As " +
        "complexity rose from FD12 to FD16, the average number of single " +
        "words submitted per trial decreased monotonically from 1.69 to " +
        "1.56 (Friedman χ² = 46.5, " },
  { t: "P", italic: true },
  { t: " = 7.9 × 10" },
  { t: "−11", sup: true },
  { t: "; Fig. 1), and the parallel decrease in the number of free-text " +
        "descriptions per trial was equally robust (" },
  { t: "P", italic: true },
  { t: " = 6.7 × 10" },
  { t: "−10", sup: true },
  { t: "). The within-subject median semantic spread of percepts at each " +
        "FD, a participant-level index of how semantically scattered the " +
        "answers were on stimuli of a given complexity, also decreased " +
        "slightly but reliably (" },
  { t: "P", italic: true },
  { t: " = 0.02). Mean reaction time did not differ by FD (" },
  { t: "P", italic: true },
  { t: " = 0.17). FD therefore acts as a behavioural manipulation that " +
        "biases observers toward fewer and progressively more uniform " +
        "percepts." },
]));
children.push(...fig(
  "fig1_fd_behaviour.png",
  "Fig. 1. Image complexity reduces the amount of pareidolia.",
  "Within-subject mean ± 95% CI by FD level for reaction time, single " +
  "words per trial, descriptions per trial, and the within-subject median " +
  "semantic spread of percepts. Brackets denote significant pair-wise " +
  "Wilcoxon signed-rank contrasts."
));

children.push(H3("Image-statistics features predict per-image pareidolia rate."));
children.push(P(
  "Beyond the discrete FD12/FD14/FD16 contrast, individual stimuli " +
  "within the same FD level varied widely in how often participants " +
  "reported a pareidolic percept (per-image rate range 0.50 to 0.95, " +
  "median 0.76). To ask which low-level image statistics drive this " +
  "variability, we extracted 24 features from each of the 300 stimuli, " +
  "spanning intensity statistics (mean, SD, skewness, kurtosis, pixel " +
  "entropy), the radially averaged power-spectrum slope and low / mid / " +
  "high frequency band shares, spatial structure (Sobel edge magnitude, " +
  "local-contrast mean and SD, count and size statistics of connected " +
  "components binarised at mean intensity), and five symmetry measures " +
  "(left-right and top-bottom pixel symmetry, gradient-orientation " +
  "symmetry, coarse-downsampled symmetry, and Fourier-magnitude " +
  "symmetry; Materials and Methods)."
));
children.push(R([
  { t: "Two image properties stood out as positive correlates of " +
        "pareidolia rate (Fig. 2). The power-spectrum slope (steeper " +
        "slope = stronger low-frequency dominance) was the single " +
        "strongest positive predictor (" },
  { t: "r", italic: true },
  { t: " = +0.37, " },
  { t: "P", italic: true },
  { t: " = 2.5 × 10" },
  { t: "−11", sup: true },
  { t: "), and the low-frequency band share showed the same effect (" },
  { t: "r", italic: true },
  { t: " = +0.34, " },
  { t: "P", italic: true },
  { t: " = 2.1 × 10" },
  { t: "−9", sup: true },
  { t: "). Symmetry contributed in the same direction: top-bottom pixel " +
        "symmetry, left-right pixel symmetry, Fourier-magnitude " +
        "symmetry, and gradient-orientation symmetry all correlated " +
        "positively with pareidolia rate (" },
  { t: "r", italic: true },
  { t: " = +0.17 to +0.24, all " },
  { t: "P", italic: true },
  { t: " < 0.005). On the other side, fragmented and high-contrast " +
        "images suppressed pareidolia: Sobel edge SD, local-contrast " +
        "SD, local-contrast mean, edge mean, and the count of connected " +
        "components were the most negative predictors (" },
  { t: "r", italic: true },
  { t: " = −0.35 to −0.38, all " },
  { t: "P", italic: true },
  { t: " < 10" },
  { t: "−9", sup: true },
  { t: "). A random-forest regression on the full feature set (5-fold " +
        "CV) captured a small but reliable amount of the per-image " +
        "variance in pareidolia rate (" },
  { t: "R", italic: true },
  { t: "2", sup: true },
  { t: " = 0.08 ± 0.07), with permutation feature importance again " +
        "putting symmetry (top-bottom and gradient L-R) and the " +
        "spectral slope at the top. Qualitatively, the highest-" +
        "pareidolia images are smooth, low-frequency FD12 silhouettes " +
        "that elicit canonical percepts (" },
  { t: "bear, dog, face, rabbit, fish", italic: true },
  { t: "), whereas the lowest-pareidolia images are dense, high-" +
        "frequency FD16 textures that fragment into many small " +
        "components and elicit no dominant percept. The same low-level " +
        "features therefore explain a substantial share of the FD " +
        "effect on the " },
  { t: "amount", italic: true },
  { t: " of pareidolia reported in Fig. 1, and identify a specific " +
        "image profile (smooth, low-frequency-dominated, symmetric) " +
        "that maximally engages the constructive side of vision." },
]));
children.push(...fig(
  "fig2_image_features.png",
  "Fig. 2. Image-statistics features predicting per-image pareidolia.",
  "Pearson correlation between each of 24 spectral, spatial, and " +
  "symmetry features (rows) and four image-level pareidolia metrics " +
  "(columns: pareidolia rate, mean words per trial, agreement on the " +
  "modal word, and per-image word diversity), across the 300 stimuli. " +
  "Cells are annotated with the signed r plus a significance marker " +
  "(* P < 0.05, ** P < 0.01, *** P < 0.001). Rows sorted by their " +
  "correlation with pareidolia rate."
));

children.push(H3("Image complexity reshapes the content of pareidolia."));
children.push(R([
  { t: "The drop in amount was accompanied by a striking shift in semantic " +
        "content. We assigned every unique percept word to one of fifteen " +
        "seed-based semantic buckets (face, body-part, person, mammal, " +
        "bird, fish-aquatic, insect-bug, reptile, creature, object, " +
        "landscape, weather, abstract, food, action) by nearest seed-" +
        "centroid in BERT space (Materials and Methods), with a hard-" +
        "reject list preventing high-frequency objects (e.g., " },
  { t: "hat, cross, hammer, gun", italic: true },
  { t: ") and actions (e.g., " },
  { t: "kiss, dancing", italic: true },
  { t: ") from leaking into the human or animal categories. Three highly " +
        "significant within-subject FD effects emerged on what " +
        "participants named (Fig. 3). The proportion of percepts falling " +
        "in human-related categories (face, body-part, person) dropped " +
        "monotonically from 33% at FD12 to 25% at FD16 " +
        "(χ² = 73.3, " },
  { t: "P", italic: true },
  { t: " = 1.3 × 10" },
  { t: "−15", sup: true },
  { t: "); the animal share (mammal, bird, fish, insect, reptile) rose " +
        "from 25% to a peak of 32% at FD14 (" },
  { t: "P", italic: true },
  { t: " = 1.2 × 10" },
  { t: "−7", sup: true },
  { t: "); and the animal-minus-human bias index inverted sign across " +
        "the FD axis, from clearly human dominated at FD12 " +
        "(−0.16) to mildly animal dominated at FD16 (+0.09; " },
  { t: "P", italic: true },
  { t: " = 6.8 × 10" },
  { t: "−13", sup: true },
  { t: ")." },
]));

children.push(R([
  { t: "At the image rather than the participant level, the same " +
        "complexity gradient produced a parallel rise in inter-observer " +
        "disagreement. The Shannon entropy of the percept-word " +
        "distribution per stimulus rose from a median of 5.90 bits at " +
        "FD12 to 6.05 bits at FD16 (Kruskal-Wallis " },
  { t: "P", italic: true },
  { t: " = 0.014; Fig. 3, rightmost panel), indicating that high-" +
        "complexity images do not converge on a canonical answer the way " +
        "low-complexity images do. The lexical shift is visible in the " +
        "most distinctive words per FD: FD12 percepts are dominated by " +
        "discrete, often anthropomorphic figures (" },
  { t: "vase, handshake, elbow, horn, pregnant, ape, cave", italic: true },
  { t: "); FD14 by mammalian and small-animal labels (" },
  { t: "ox, warthog, bunnies, salamander, llama, mice, rooster",
    italic: true },
  { t: "); FD16 by texture and atmosphere descriptors (" },
  { t: "canopy, stars, fuzzy, marble, rain, scatter, noise, swamp",
    italic: true },
  { t: "). The progression from figural to animal to textural is also " +
        "visible spatially in a t-SNE projection of every unique percept " +
        "word coloured by majority FD level (Fig. 4b): FD12 words " +
        "concentrate in face and person regions, FD14 in the central " +
        "animal cloud, and FD16 spreads into the abstract and textural " +
        "periphery." },
]));
children.push(...fig(
  "fig3_fd_perception.png",
  "Fig. 3. Image complexity reshapes the content of pareidolia.",
  "Within-subject mean ± 95% CI per FD of the human share, animal " +
  "share, and animal-minus-human bias index (panels 1 to 3, bars), together " +
  "with the per-image vocabulary entropy of percept distributions " +
  "(rightmost panel, bar = mean across the ten images per FD). Brackets " +
  "denote significant pair-wise contrasts (Wilcoxon for the participant-" +
  "level metrics, Mann-Whitney U for the image-level panel)."
));
children.push(...fig(
  "fig4_semantic_territory.png",
  "Fig. 4. Semantic territory of collective pareidolia.",
  "Shared t-SNE projection (cosine, perplexity = 30) of every unique " +
  "percept word (n = 4,361); dot size scales with corpus frequency; KDE " +
  "contours per group. (a) coloured by majority creativity tertile " +
  "(sequential warm palette, amber = low, dark red = high DAT). (b) " +
  "coloured by majority FD level at which the word was produced. Both " +
  "panels share the same projection.",
  430
));

children.push(H3("Observers match fractal dimension to a percept " +
  "family's spatial scale."));
children.push(P(
  "To complement the fifteen-bucket taxonomy of Fig. 3 with a finer " +
  "view oriented around the spatial scale of the meanings extracted, " +
  "we hand-curated eight hierarchically organised percept families " +
  "spanning the most frequent vocabulary in the corpus: face / body " +
  "part, person / people, animal, mythical creature, plant, " +
  "place / landscape, object / symbol, and abstract / texture " +
  "(Materials and Methods). Coverage on the corpus of valid percept " +
  "tokens was 59% (the residual is largely geography terms, foods, and " +
  "rare creatures unique to a single observer); per-FD report rates " +
  "within the covered fraction were z-scored across the three FD " +
  "conditions to expose the shape of each family's tuning."
));
children.push(R([
  { t: "Two converging patterns emerge (Fig. 5). First, families " +
        "differ sharply in " },
  { t: "where", italic: true },
  { t: " on the FD axis they peak. Faces, body parts, persons, places, " +
        "and discrete objects are all over-reported on the smoothest " +
        "stimuli (FD ≈ 1.3; " },
  { t: "z", italic: true },
  { t: " = +0.7 to +1.4) and under-reported on the busiest ones " +
        "(FD ≈ 1.7; " },
  { t: "z", italic: true },
  { t: " = −0.9 to −1.4). Animals peak at the middle complexity level " +
        "(FD ≈ 1.5; " },
  { t: "z", italic: true },
  { t: " = +0.8), as do mythical creatures (" },
  { t: "z", italic: true },
  { t: " = +0.4 at FD 1.5 and " },
  { t: "z", italic: true },
  { t: " = +0.9 at FD 1.7). Plants and abstract or texture descriptors " +
        "peak strongly on the busiest stimuli (FD ≈ 1.7; " },
  { t: "z", italic: true },
  { t: " = +1.4 in both cases). Second, computing each family's rate-" +
        "weighted preferred FD (a continuous summary of its tuning) " +
        "orders the families on a smooth-to-rough axis from " +
        "approximately 1.45 (persons, places, faces, objects) through " +
        "1.49 (animals) to 1.54 (mythical creatures, abstract textures, " +
        "plants), recovering the expected progression from global, " +
        "coherent forms at low complexity to fine, repeated structure " +
        "at high complexity. The same gradient is visible at the " +
        "percept level (Fig. 5b): the 99 individual words reported on " +
        "at least 30 trials span a preferred-FD range of roughly 1.31 " +
        "to 1.62, with the family means falling along the heatmap-" +
        "derived ordering. The result formalises the qualitative " +
        "content shift documented in Fig. 3 as a continuous mapping " +
        "between stimulus complexity and the spatial scale of the " +
        "meanings observers extract." },
]));
children.push(...fig(
  "fig5_fd_family.png",
  "Fig. 5. Observers match fractal dimension to a percept family's " +
  "spatial scale.",
  "(a) For each of eight hand-curated percept families (rows) and each " +
  "FD level (columns), the relative report rate z-scored across FD " +
  "within family; red = the family peaks at this FD. Rows ordered by " +
  "each family's rate-weighted preferred FD, smooth (top) to rough " +
  "(bottom). (b) Preferred FD per individual percept word with at " +
  "least 30 reports (dots; mean FD of images where the word was " +
  "reported), grouped by family with the family mean shown as a " +
  "horizontal bar. Family coverage is 59% of all valid percept tokens.",
  500
));

children.push(H3("Observers largely disagree on what they see, but " +
  "agreement decreases with complexity."));
children.push(R([
  { t: "Beyond the group-level shifts in modal percept, we asked how " +
        "much individual observers actually converge on the same answer " +
        "for the same image. For every stimulus (300 images, " +
        "approximately 50 observers each) we computed two complementary " +
        "inter-observer agreement metrics: the " },
  { t: "modal-word share", italic: true },
  { t: ", defined as the proportion of all percept words on that image " +
        "that equal the single most common word, and the " },
  { t: "mean cross-observer cosine similarity", italic: true },
  { t: ", defined as the mean BERT cosine similarity between two " +
        "randomly sampled percept words from " },
  { t: "different", italic: true },
  { t: " participants on the same image (the literal same word yields " +
        "1.0; close synonyms yield ~0.6 to 0.8; unrelated words yield " +
        "~0.1). Both metrics agree that consensus is low in absolute " +
        "terms: on average the single most common word accounts for " +
        "only 7% of all percepts (best image: 27%; worst image: 4%), " +
        "and the mean cross-observer cosine similarity sits around " +
        "0.29 (Fig. 6). Pareidolia, at this resolution, is mostly an " +
        "idiosyncratic act. The small but reliable consensus core that " +
        "does exist is modulated by image complexity: low-FD images " +
        "elicit higher agreement than high-FD images on both metrics " +
        "(Kruskal-Wallis on mean cosine " },
  { t: "P", italic: true },
  { t: " = 0.004; on modal share " },
  { t: "P", italic: true },
  { t: " = 4 × 10" },
  { t: "−5", sup: true },
  { t: "), with monotonic decreases across FD12 → FD14 → FD16. A " +
        "complementary per-(image, DAT-tertile) analysis paired across " +
        "the 300 images found no detectable effect of observer " +
        "creativity on per-image consensus (Friedman " },
  { t: "P", italic: true },
  { t: " = 0.70 on mean cosine, " },
  { t: "P", italic: true },
  { t: " = 0.22 on vocabulary entropy), indicating that the ambiguity " +
        "of an image is a property of the image itself rather than of " +
        "the creativity profile of its observers." },
]));
children.push(...fig(
  "fig6_image_consensus.png",
  "Fig. 6. Inter-observer agreement decreases with image complexity.",
  "Per-image inter-observer agreement summarised across 300 stimuli " +
  "(100 per FD level). (a) Mean cross-observer cosine similarity " +
  "between two randomly sampled percept embeddings from different " +
  "participants on the same image. (b) Modal-word share: fraction of " +
  "all percept words on the image that equal the single most common " +
  "word. Box-and-whisker plots show the distribution across the 100 " +
  "images per FD level; dots are individual images. Brackets denote " +
  "significant pair-wise Mann-Whitney U contrasts.",
  430
));

children.push(H3("Test-retest reliability of DAT is preserved across the task."));
children.push(R([
  { t: "Among the " },
  { t: "n", italic: true },
  { t: " = 507 participants who completed both DAT timepoints (Fig. 7), " +
        "pre- and post-task scores were positively but moderately " +
        "correlated (Pearson " },
  { t: "r", italic: true },
  { t: " = 0.19, " },
  { t: "P", italic: true },
  { t: " = 1.2 × 10" },
  { t: "−5", sup: true },
  { t: "), with no detectable group-level shift (Wilcoxon signed-rank " },
  { t: "W", italic: true },
  { t: " = 6.1 × 10" },
  { t: "4", sup: true },
  { t: ", " },
  { t: "P", italic: true },
  { t: " = 0.99; mean Δ = −0.02, SD = 4.4). The link between DAT " +
        "and pareidolia reported below replicates consistently with the " +
        "pre-task score and weakens substantially when the post-task " +
        "score is substituted in, consistent with this link capturing a " +
        "stable trait-level property of the observer rather than a " +
        "transient state shift induced by the pareidolia task." },
]));
children.push(...fig(
  "fig7_dat_pre_post.png",
  "Fig. 7. Pre vs post DAT scores.",
  "Left: GloVe-scored DAT post vs DAT pre, n = 507. Right: distribution " +
  "of the participant-level delta (post minus pre).",
  360
));

children.push(H3("Trait creativity does not predict more or faster pareidolia."));
children.push(R([
  { t: "Across six general behavioural metrics (fraction of trials " +
        "completed, single words per trial, descriptions per trial, mean " +
        "reaction time, total unique words across the experiment, and " +
        "mean corpus rarity of a participant’s words), only one " +
        "survived bivariate ±3 SD outlier trimming as significantly " +
        "related to DAT, and in the opposite direction to a “more " +
        "is more” account: high-creativity participants completed " +
        "slightly " },
  { t: "fewer", italic: true },
  { t: " trials than low-creativity participants (" },
  { t: "r", italic: true },
  { t: " = −0.10, " },
  { t: "P", italic: true },
  { t: " = 0.03; Fig. 8). Words per trial and reaction time were both " +
        "null. Creativity, in this dataset, therefore does not manifest " +
        "as producing more or faster pareidolia." },
]));
children.push(...fig(
  "fig8_dat_behaviour.png",
  "Fig. 8. Creativity is not associated with more or faster pareidolia.",
  "DAT vs mean words per trial (left) and DAT vs mean reaction time " +
  "(right). The number of percepts extracted per trial and the time " +
  "taken to extract them do not covary with creativity.",
  360
));

children.push(H3("Creativity predicts greater perceptual diversity, " +
  "concentrated at intermediate complexity."));
children.push(R([
  { t: "The within-subject median pair-wise cosine distance of a " +
        "participant’s percept embeddings, a participant-level " +
        "measure of how semantically scattered the percepts are, was " +
        "positively correlated with DAT (" },
  { t: "r", italic: true },
  { t: " = +0.14, " },
  { t: "P", italic: true },
  { t: " = 0.001; " },
  { t: "n", italic: true },
  { t: " = 500; Fig. 9, left), reproducing a previously reported " +
        "relationship between divergent thinking and lexical diversity in " +
        "a perceptual task. Splitting the analysis by FD revealed that " +
        "the effect concentrates at the middle complexity level " +
        "(Fig. 9, right): " },
  { t: "r", italic: true },
  { t: " = +0.17 (" },
  { t: "P", italic: true },
  { t: " = 2.7 × 10" },
  { t: "−4", sup: true },
  { t: ") at FD14, " },
  { t: "r", italic: true },
  { t: " = +0.09 (" },
  { t: "P", italic: true },
  { t: " = 0.04) at FD12, and essentially null at FD16 (" },
  { t: "r", italic: true },
  { t: " ≈ 0, " },
  { t: "P", italic: true },
  { t: " = 0.93). Intermediate complexity therefore appears to be the " +
        "regime in which divergent-thinking ability has the greatest " +
        "expressive room." },
]));
children.push(...fig(
  "fig9_diversity.png",
  "Fig. 9. Creativity predicts perceptual diversity, especially at " +
  "intermediate complexity.",
  "(left) pooled across all FD levels (one point per participant). " +
  "(right) the same analysis split by FD: the relationship concentrates " +
  "at FD14, the middle complexity level."
));

children.push(H3("Creativity biases pareidolia from humans toward animals."));
children.push(R([
  { t: "The same fifteen-bucket taxonomy pinpointed where the link " +
        "between DAT and diversity comes from. The per-participant share " +
        "of percepts falling in human categories (face + body-part + " +
        "person) was negatively associated with DAT (" },
  { t: "r", italic: true },
  { t: " = −0.14, " },
  { t: "P", italic: true },
  { t: " = 0.003), while the animal share was numerically higher in " +
        "high-creativity participants but did not reach significance on " +
        "its own (" },
  { t: "r", italic: true },
  { t: " = +0.08, " },
  { t: "P", italic: true },
  { t: " = 0.10; Fig. 10). Combining the two into an animal-minus-human " +
        "bias index produced the cleanest creativity effect we observed (" },
  { t: "r", italic: true },
  { t: " = +0.12, " },
  { t: "P", italic: true },
  { t: " = 0.006; Spearman ρ = +0.14, " },
  { t: "P", italic: true },
  { t: " = 0.001). High-creativity participants therefore do not extract " },
  { t: "more", italic: true },
  { t: " pareidolic percepts overall. Rather, they shift the " },
  { t: "kind", italic: true },
  { t: " of percept they extract, away from canonical human and face " +
        "responses and toward animal ones." },
]));
children.push(...fig(
  "fig10_human_animal_bias.png",
  "Fig. 10. Creativity tilts the percept distribution away from humans " +
  "and toward animals.",
  ""
));

children.push(H3("Unsupervised clusters separate low- and high-creativity " +
  "profiles."));
children.push(R([
  { t: "The human-vs-animal contrast captures one axis along which " +
        "creativity restructures percepts, but the seed-based taxonomy " +
        "is deliberately narrow. To examine the full preference " +
        "landscape we re-derived categories " },
  { t: "de novo", italic: true },
  { t: " by clustering all percept words that occurred at least three " +
        "times in the corpus (Materials and Methods). HDBSCAN yielded " +
        "32 interpretable clusters plus a small noise group (22% of " +
        "words); about a third of clusters had a 95% bootstrap CI on " +
        "the log-odds (high vs low tertile) preference that excluded " +
        "zero (Fig. 11b)." },
]));
children.push(R([
  { t: "High-creativity participants were over-represented in clusters of " },
  { t: "specific, embodied, or unusual", italic: true },
  { t: " percepts: " },
  { t: "mushroom / apple / clover / carrot", italic: true },
  { t: "; " },
  { t: "gremlin / ogre / gargoyle / ghoul", italic: true },
  { t: "; " },
  { t: "blood / stomach / kidney / vomit", italic: true },
  { t: "; " },
  { t: "fish / seahorse / crab / whale", italic: true },
  { t: "; " },
  { t: "bat / vase / ball / cob", italic: true },
  { t: "; " },
  { t: "hand / finger / fist / hands", italic: true },
  { t: "; " },
  { t: "witch / cartoon / demon / angel", italic: true },
  { t: ". Low-creativity participants were over-represented in clusters of " },
  { t: "abstract or texture-like", italic: true },
  { t: " percepts: " },
  { t: "plus / separation / two / open", italic: true },
  { t: "; " },
  { t: "water / spill / mess / pipe", italic: true },
  { t: "; " },
  { t: "ink / paint / painting / book", italic: true },
  { t: "; " },
  { t: "ocean / lake / river / sea", italic: true },
  { t: "; " },
  { t: "land / mountain / cliff", italic: true },
  { t: "; " },
  { t: "fire / explosion / death", italic: true },
  { t: "; " },
  { t: "hammer / hole / splatter / dots", italic: true },
  { t: ". The semantic-map view (Fig. 11a) makes the structure of the " +
        "contrast explicit: the two preference profiles occupy " +
        "non-overlapping regions of BERT space, with the creature, " +
        "viscera, and specific-object region on the upper side and the " +
        "landscape, texture, and abstract region on the lower side. " +
        "Together with the bias-index result above, this argues that " +
        "the creativity effect on pareidolia is best characterised not " +
        "as seeing " },
  { t: "more", italic: true },
  { t: " but as seeing " },
  { t: "different", italic: true },
  { t: " things." },
]));
children.push(...fig(
  "fig11_preferred_categories.png",
  "Fig. 11. Percept clusters preferred by low vs high creativity.",
  "(a) 2D UMAP projection of every percept word with at least 3 corpus " +
  "occurrences, coloured by the log-odds preference of its HDBSCAN " +
  "cluster (red = preferred by high-creativity participants, amber = " +
  "preferred by low-creativity). Labels mark the top six clusters on " +
  "each side at their UMAP centroid. (b) Forest plot of all 32 retained " +
  "clusters, log-odds (high/low) with bootstrap 95% CI; stars mark " +
  "clusters whose CI excludes zero. Numbers on the right are word " +
  "occurrences per cluster."
));

children.push(H3("The same pipeline reveals a complementary FD preference axis."));
children.push(R([
  { t: "Repurposing the clustering with stimulus FD level as the " +
        "contrast variable (FD16 vs FD12 instead of high vs low DAT) " +
        "shows partly overlapping but distinct preference geometries " +
        "(Fig. 12). High-FD images are over-represented in clusters of " },
  { t: "tree / forest", italic: true },
  { t: ", " },
  { t: "butterfly / caterpillar / worm", italic: true },
  { t: ", " },
  { t: "alien / sky / space", italic: true },
  { t: ", " },
  { t: "fish / seahorse / crab", italic: true },
  { t: ", " },
  { t: "witch / demon / angel", italic: true },
  { t: ", and " },
  { t: "ghost / lamp / chaos", italic: true },
  { t: ", whereas low-FD images dominate clusters of " },
  { t: "cave / rock / stones", italic: true },
  { t: ", " },
  { t: "blood / stomach / kidney", italic: true },
  { t: ", " },
  { t: "boot / foot / arm", italic: true },
  { t: ", " },
  { t: "bat / vase / ball", italic: true },
  { t: ", and " },
  { t: "land / mountain / cliff", italic: true },
  { t: ". Creativity drives a humans-to-animals shift along one axis of " +
        "the BERT space; image complexity drives a body-parts-to-" +
        "atmospheric-creatures shift along another. The two manipulations " +
        "therefore probe partly independent dimensions of the same " +
        "percept landscape." },
]));
children.push(...fig(
  "fig12_preferred_categories_fd.png",
  "Fig. 12. Percept clusters preferred by low vs high stimulus FD.",
  "Same clustering pipeline as Fig. 11; the contrast variable is FD16 " +
  "versus FD12. (a) Semantic map coloured by log-odds preference " +
  "(blue = FD12, red = FD16). (b) Forest plot of clusters, log-odds " +
  "(FD16/FD12) with bootstrap 95% CI; stars mark clusters whose CI " +
  "excludes zero."
));

children.push(H3("Gaze signatures of successful pareidolia."));
children.push(P(
  "For the 247 participants who opted in to webcam-based gaze recording " +
  "and produced trials that survived the WebGazer tracking-failure check " +
  "(Materials and Methods), we computed six gaze metrics per trial " +
  "spanning fixation structure (count, mean duration), scanpath length, " +
  "and three spatial-concentration measures (gaze entropy in bits, Gini " +
  "coefficient of the 2D gaze histogram, and recurrence rate of nearby " +
  "fixations). We contrasted these metrics along three independent axes: " +
  "trials on which the participant did vs did not report a pareidolic " +
  "percept, stimulus FD level, and continuous DAT score (Fig. 13)."
));
children.push(R([
  { t: "Within the 105 participants who had at least two trials of each " +
        "kind, trials in which the participant submitted at least one " +
        "word (“pareidolia”) differed from no-word trials on " +
        "three of the six metrics, all in the same direction: more " +
        "fixations per trial (9.0 vs 9.4, Wilcoxon " },
  { t: "P", italic: true },
  { t: " = 0.045); lower 2D gaze entropy (3.76 vs 3.71 bits, " },
  { t: "P", italic: true },
  { t: " = 0.037); and a higher Gini concentration of the gaze " +
        "distribution (0.868 vs 0.872, " },
  { t: "P", italic: true },
  { t: " = 0.047). Fixation duration, scanpath length, and recurrence " +
        "rate did not differ. Successful pareidolia is therefore " +
        "accompanied by a small but reliable shift toward " },
  { t: "more", italic: true },
  { t: " fixations clustered in a " },
  { t: "smaller", italic: true },
  { t: " region, consistent with the participant having found something " +
        "specific in the noise that they are inspecting." },
]));
children.push(R([
  { t: "Within-subject FD effects emerged on scanpath length (" },
  { t: "P", italic: true },
  { t: " = 0.006; monotonic decrease from FD12 to FD16) and, marginally, " +
        "on fixation duration (" },
  { t: "P", italic: true },
  { t: " = 0.06; longer fixations at FD16). The other four metrics were " +
        "not modulated by FD. As image complexity rises, participants " +
        "therefore move their eyes less and dwell on a few central " +
        "locations for longer, consistent with the behavioural and " +
        "content-level FD effects in Figs. 1 and 2. None of the six " +
        "metrics correlated reliably with DAT score (|" },
  { t: "r", italic: true },
  { t: "| < 0.07 for all six; all " },
  { t: "P", italic: true },
  { t: " > 0.30). Together with the behavioural-engagement null reported " +
        "above, this argues that creativity does not change " },
  { t: "how", italic: true },
  { t: " people look at the stimulus, only " },
  { t: "what", italic: true },
  { t: " they extract from it." },
]));
children.push(...fig(
  "fig13_eye_tracking.png",
  "Fig. 13. Eye-tracking metrics across three contrasts.",
  "Rows: (A) pareidolia vs no-pareidolia trials (within-subject Wilcoxon, " +
  "n = 105 participants with at least 2 trials of each kind); (B) " +
  "stimulus FD level (within-subject Friedman plus pair-wise Wilcoxon, " +
  "n = 226); (C) continuous DAT score (between-subject Pearson, n = 198 " +
  "to 227 depending on the metric, ±3 SD bivariate trim). Columns: " +
  "six gaze metrics mixing classical (fixation count, fixation duration, " +
  "scanpath length, gaze entropy) and recent measures (Gini concentration " +
  "of the 2D gaze histogram, RQA-style recurrence rate of nearby " +
  "fixations). Brackets mark significant pair-wise contrasts; small " +
  "in-axis text on row C reports the Pearson r."
));

// ── DISCUSSION ──────────────────────────────────────────────────────────────
children.push(H2("Discussion"));

children.push(P(
  "Three findings emerge consistently across analyses. First, the " +
  "statistical complexity of the noise determines the modal pareidolic " +
  "percept on a human-to-animal-to-texture continuum, with the " +
  "between-image variability of percepts growing as complexity rises. " +
  "Second, trait creativity, as indexed by the DAT, does not modify how " +
  "much pareidolia an observer produces but shifts which categories the " +
  "observer favours, away from canonical human and face percepts and " +
  "toward animals, viscera, and specific embodied objects. Third, both " +
  "manipulations leave converging gaze signatures: successful pareidolia " +
  "is accompanied by more concentrated fixations, and rising image " +
  "complexity tightens the scan pattern, while creativity has no " +
  "detectable gaze signature."
));
children.push(P(
  "The asymmetry between the rich content effects of creativity and its " +
  "null behavioural and oculomotor effects is informative. It suggests " +
  "that what divergent-thinking measures track in this paradigm is not a " +
  "bias toward sensitivity (more percepts, faster percepts) but a bias " +
  "toward unusual readout of an otherwise comparable extraction process. " +
  "A practical consequence is that pareidolia tasks designed to maximise " +
  "individual differences should score the kind of percept rather than " +
  "its presence or its count. A second consequence is that mid-complexity " +
  "stimuli (FD14 in our parameterisation) are the regime in which " +
  "observer-driven variability is most expressible."
));
children.push(P(
  "The pre versus post DAT result, near-zero mean shift but moderate " +
  "test-retest correlation, supports the interpretation of DAT as a " +
  "stable trait rather than a state, and suggests that completing " +
  "approximately 15 minutes of pareidolia does not measurably alter " +
  "divergent-thinking ability. The strong attenuation of links between " +
  "post-task DAT and pareidolia, alongside the preserved links to " +
  "pre-task DAT, further supports a trait reading."
));
children.push(P(
  "Several limitations should be noted. Webcam eye tracking, although " +
  "adequate for the coarse-grained contrasts reported here, lacks the " +
  "precision to resolve fixation-level effects within the image; the " +
  "moderate-strength gaze effects we report are unlikely to be " +
  "overestimates. The semantic categorisation rests on contemporary " +
  "sentence embeddings and inherits whatever biases those embeddings " +
  "carry; we mitigated this by combining a seed-based taxonomy with an " +
  "unsupervised HDBSCAN view, and the two views agreed on the principal " +
  "contrasts. Finally, the design samples three discrete FD levels rather " +
  "than a continuum; a finer sweep would test the monotonicity claims " +
  "more strictly."
));

// ── METHODS ─────────────────────────────────────────────────────────────────
children.push(H2("Materials and Methods"));

children.push(H3("Participants."));
children.push(R([
  { t: "Participants were recruited online via [recruitment platform]; " +
        "all gave informed consent in accordance with the [ethics " +
        "committee] guidelines. The full session was completed by " },
  { t: "N", italic: true },
  { t: " = 579 English-speaking adults (the “main cohort”), " +
        "comprising a demographic form, the 10-word Divergent Association " +
        "Task (Olson et al., 2021) as a baseline measure of trait " +
        "creativity (DAT pre), an optional 9-point WebGazer calibration " +
        "and validation routine, 30 pareidolia trials with free-response " +
        "word entry, a repeated DAT (DAT post; " },
  { t: "n", italic: true },
  { t: " = 507 provided a scoreable response), and a closing feedback " +
        "questionnaire. The main cohort analysed below is defined as the " +
        "intersection of (i) English as primary language, (ii) feedback " +
        "supplied, and (iii) at least one valid pareidolia trial. " +
        "Sessions whose stored DAT word list duplicated one of the " +
        "previous two log entries were removed." },
]));

children.push(H3("Stimuli."));
children.push(R([
  { t: "The stimulus set comprised 30 greyscale fractal-noise images, " +
        "ten at each of three fractal-dimension levels labelled " },
  { t: "FD12, FD14, FD16", bold: true },
  { t: " by their generation target. Each image was synthesised by " +
        "filtering 2D Gaussian noise so that its amplitude spectrum " +
        "followed a 1/" },
  { t: "f", italic: true },
  { t: "β profile prior to inverse Fourier transform; higher FD " +
        "produces finer-grained spatial detail and lower local " +
        "autocorrelation. The realised fractal dimension of every " +
        "generated image was estimated empirically with the standard 2D " +
        "box-counting method (Liebovitch and Toth, 1989); across the 30 " +
        "images the mean ± SD per nominal level was " },
  { t: "FD12: 1.27 ± 0.012, FD14: 1.49 ± 0.016, " +
        "FD16: 1.67 ± 0.011", bold: true },
  { t: " (100 images per level were generated and the 10 closest to each " +
        "target were retained). For readability we refer to the three " +
        "conditions throughout the manuscript as FD ≈ 1.3, 1.5, and " +
        "1.7. Each participant saw all 30 images in a fully randomised " +
        "order, each rendered centred on a white viewport with " },
  { t: "max-width: 55%", code: true },
  { t: "." },
]));

children.push(H3("Pareidolia task."));
children.push(R([
  { t: "A pareidolia trial began with a 500 ms central fixation cross, " +
        "followed by the stimulus image presented for up to 30 s, " +
        "terminated early if the participant pressed the spacebar to " +
        "advance. Immediately after stimulus offset a free-text response " +
        "screen appeared with five short-answer slots labelled " +
        "“Single words” and five longer slots labelled " +
        "“Descriptions (optional)”; participants were " +
        "instructed to write every distinct percept that the image " +
        "evoked. When eye tracking had been enabled at session start, " +
        "raw (" },
  { t: "x, y, t", italic: true },
  { t: ") gaze samples were captured throughout image presentation via " +
        "the WebGazer browser-side extension to jsPsych." },
]));

children.push(H3("Percept word preprocessing."));
children.push(R([
  { t: "Each typed entry was lowercased and whitespace-stripped. To " +
        "minimise contamination by non-percept entries we kept only " +
        "entries matching the regular expression " },
  { t: "^[A-Za-z]+$", code: true },
  { t: " (single alphabetic tokens without spaces, digits, or " +
        "punctuation) and additionally removed the catch-all stopwords " },
  { t: "nothing, nothingness, black, white, map, cloud, clouds",
    italic: true },
  { t: ". Sentence-level descriptions were excluded from semantic " +
        "analyses to avoid mixing single-word and multi-word vector " +
        "representations, which occupy systematically different regions " +
        "of embedding space. All semantic computations use 384-" +
        "dimensional sentence-transformer embeddings (" },
  { t: "all-MiniLM-L6-v2", code: true },
  { t: ") of every unique surviving percept word (" },
  { t: "n", italic: true },
  { t: "words", sub: true },
  { t: " = 4,361)." },
]));

children.push(H3("Divergent Association Task (DAT) scoring."));
children.push(R([
  { t: "Both DAT pre and DAT post were scored with the canonical Olson " +
        "et al. (2021) procedure. Each set of 10 candidate words was " +
        "lowercased and restricted to alphabetic forms; the first seven " +
        "words with a GloVe 840B 300-d vector were retained, and the " +
        "DAT score was defined as 100 × mean(" },
  { t: "d", italic: true },
  { t: "ij", sub: true, italic: true },
  { t: "), where " },
  { t: "d", italic: true },
  { t: "ij", sub: true, italic: true },
  { t: " is the cosine distance between every pair of those seven " +
        "vectors. Sessions producing fewer than seven valid lookups " +
        "yielded no DAT score and were excluded from any analysis using " +
        "that timepoint. DAT scores in main-text analyses refer to the " +
        "baseline (pre) score unless explicitly noted." },
]));

children.push(H3("Eye-tracking preprocessing."));
children.push(R([
  { t: "The online WebGazer calibration sweep provides a dense set of " +
        "gaze samples spanning most of the viewport. For each session, " +
        "we estimated the usable gaze extent as the 1st-to-99th-" +
        "percentile box of these calibration samples and used it to " +
        "rescale all subsequent gaze data into the unit square [0, 1], " +
        "eliminating cross-session differences in screen size and " +
        "browser viewport. The stimulus region of interest (ROI) was " +
        "approximated as a per-session rectangle centred on the recorded " +
        "stimulus position with half-width 0.275 (half of the " },
  { t: "55%", code: true },
  { t: " CSS rule) and half-height 0.275 · " },
  { t: "W/H", italic: true },
  { t: ", where " },
  { t: "W/H", italic: true },
  { t: " is the session’s viewport aspect ratio. Trials whose " +
        "normalised gaze trace formed a near-perfect line, defined as " +
        "|corr(" },
  { t: "x, y", italic: true },
  { t: ")| > 0.97 together with span > 0.6 on both axes, a signature " +
        "of the WebGazer face-mesh tracker failing onto a screen edge, " +
        "were flagged as tracking failures and dropped (530 of 5,660 " +
        "trials, 9.4%). Fixations were detected with a dispersion-" +
        "threshold (I-DT) algorithm using a dispersion bound of 0.05 in " +
        "normalised coordinates and a minimum duration of 100 ms." },
]));

children.push(H3("Per-participant and per-image semantic metrics."));
children.push(R([
  { t: "For every participant and every FD level we computed (i) the " },
  { t: "semantic spread", italic: true },
  { t: " of percepts as the median pair-wise cosine distance over their " +
        "unique word embeddings, and (ii) the " },
  { t: "vocabulary surprisal", italic: true },
  { t: " as the mean −log " },
  { t: "p", italic: true },
  { t: " over the corpus frequency of each word. To examine the " },
  { t: "kind", italic: true },
  { t: " of percept rather than its amount, we assigned every unique " +
        "word to one of fifteen semantic categories (" },
  { t: "face, body-part, person, mammal, bird, fish-aquatic, insect-bug, " +
        "reptile, creature, object, landscape, weather, abstract, food, " +
        "action", italic: true },
  { t: ") by nearest seed-centroid in BERT space (minimum cosine " +
        "≥ 0.30, otherwise “other”). Seed lists were " +
        "hand-curated and audited against the highest-frequency words " +
        "in the corpus; an explicit hard-reject list prevents a small " +
        "number of items that are close to a human or animal centroid " +
        "in embedding space but are clearly not (objects: " },
  { t: "hat, cross, hammer, gun, ball, bat, vase", italic: true },
  { t: "; actions: " },
  { t: "kiss, kissing, dancing, dance, hugging", italic: true },
  { t: "; landscape: " },
  { t: "sea, lake, ocean, river", italic: true },
  { t: ") from being assigned to those categories; such items are " +
        "routed to their best non human/animal centroid instead. For " +
        "each (participant, FD) we summarised the percept distribution " +
        "as the " },
  { t: "human share", italic: true },
  { t: " (face + body-part + person), the " },
  { t: "animal share", italic: true },
  { t: " (mammal + bird + fish + insect + reptile), and the " },
  { t: "animal-minus-human bias index", italic: true },
  { t: " (" },
  { t: "A − H", italic: true },
  { t: ") / (" },
  { t: "A + H", italic: true },
  { t: "), ranging from −1 (entirely human) to +1 (entirely " +
        "animal). At the image level we computed the Shannon entropy of " +
        "the percept-word distribution elicited by each stimulus, as a " +
        "measure of inter-observer consensus." },
]));

children.push(H3("Unsupervised clustering of percepts."));
children.push(R([
  { t: "To complement the seed-based taxonomy with an unsupervised " +
        "view, we clustered every percept word that appeared at least " +
        "three times in the corpus (" },
  { t: "n", italic: true },
  { t: " = 1,110 words). The BERT embeddings were L2 normalised, " +
        "reduced to 10 dimensions with UMAP (cosine metric, " },
  { t: "n_neighbors", code: true },
  { t: " = 15, " },
  { t: "min_dist", code: true },
  { t: " = 0), and clustered with HDBSCAN (" },
  { t: "min_cluster_size", code: true },
  { t: " = 12, " },
  { t: "cluster_selection_method", code: true },
  { t: " = EOM). The procedure produced 32 interpretable clusters plus " +
        "a noise group (22% of words). Each cluster was labelled by its " +
        "top four log-odds most distinctive words, i.e., the words " +
        "whose probability inside the cluster most exceeds their " +
        "probability outside it." },
]));

children.push(H3("Statistical analyses."));
children.push(R([
  { t: "Within-subject effects of FD were tested with Friedman’s " +
        "χ² followed by pair-wise Wilcoxon signed-rank " +
        "contrasts; only contrasts significant at " },
  { t: "P", italic: true },
  { t: " < 0.05 are annotated on the figures (brackets with asterisks). " +
        "Between-subject DAT effects were tested with Pearson’s " },
  { t: "r", italic: true },
  { t: " (reported in the main figures) and Spearman’s ρ " +
        "(reported in the companion CSV tables). Before every " +
        "correlation we trimmed bivariate outliers at ±3 SD on " +
        "both the dependent metric and the DAT score; no other " +
        "transformations were applied. For the cluster-preference forest " +
        "plot, 95% bootstrap CIs (2,000 binomial resamples per cluster) " +
        "were computed on the log-odds of the (high/low) tertile share. " +
        "Statistics are reported as " },
  { t: "r", italic: true },
  { t: " with significance markers (" },
  { t: "*", sup: true },
  { t: "P", italic: true },
  { t: " < 0.05, " },
  { t: "**", sup: true },
  { t: "P", italic: true },
  { t: " < 0.01, " },
  { t: "***", sup: true },
  { t: "P", italic: true },
  { t: " < 0.001); exact " },
  { t: "P", italic: true },
  { t: " values and Spearman counterparts are given in the supplementary " +
        "data tables. Analyses were performed in Python 3.10 using " },
  { t: "pandas", code: true }, { t: ", " },
  { t: "numpy", code: true }, { t: ", " },
  { t: "scipy.stats", code: true }, { t: ", " },
  { t: "sentence-transformers", code: true }, { t: ", " },
  { t: "umap-learn", code: true }, { t: ", and " },
  { t: "hdbscan", code: true },
  { t: "; reproducible scripts and CSV outputs are available at [URL]." },
]));

// ── build document ──────────────────────────────────────────────────────────
const doc = new Document({
  creator: "Claude",
  title: "Image statistics and trait creativity jointly shape visual pareidolia",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 34, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 0, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "333333" },
        paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 22, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 220, after: 80 }, outlineLevel: 2 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT_PATH, buf);
  console.log(`Wrote ${OUT_PATH}  (${buf.length.toLocaleString()} bytes)`);
});
