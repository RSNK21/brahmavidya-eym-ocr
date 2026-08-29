# EyM Skill Specification: Devanagari Manuscript Handwriting, Glyph-Normalisation and Context-Aware HTR

**Skill slug:** `eym-devanagari-manuscript-htr`  
**Target system:** EyM OCR hosted at `https://brahmavidya.co.in`  
**Primary script:** Devanagari  
**Primary use case:** Sanskrit and related Indic manuscript digitisation  
**Status:** Architecture and research specification  
**Date:** 2026-08-29

---

## 1. Purpose

EyM should not treat a handwritten Devanagari manuscript as a collection of isolated pixel patterns.

The intended system is a **context-aware Handwritten Text Recognition (HTR) and canonicalisation pipeline** in which:

1. the image is analysed as a physical document;
2. writing regions and text lines are segmented;
3. the writing medium and degradation are estimated;
4. the visible glyphs are interpreted against a learned model of Devanagari;
5. handwriting variation is mapped toward canonical Devanagari grapheme forms;
6. candidate readings are generated rather than blindly committing to the first visual match;
7. linguistic and orthographic context ranks the candidates;
8. a Unicode Devanagari transcription is produced;
9. uncertainty is retained;
10. the interface shows the proposed reading and, where necessary, an acceptable correction for scholarly review.

The system should therefore model:

> **image → physical/writing conditions → glyph/hand representation → grapheme sequence → linguistic candidate sequence → canonical Unicode → human-verifiable reading**

This is substantially different from conventional character-by-character OCR.

---

# 2. Important finding: there is no single GitHub repository that already does all of this

A search of current public repositories shows several useful Devanagari handwriting projects, but none should be represented as a complete solution to the EyM requirement.

The closest building blocks are:

### A. Devanagari handwriting recognition

**`subhrajyotidasgupta/DevanagariHTR`**

https://github.com/subhrajyotidasgupta/DevanagariHTR

This is explicitly a deep-learning project for detection and recognition of handwritten Devanagari/Hindi text. It is useful as a reference implementation, but it is an older project and should not be treated as the final EyM model.

### B. Devanagari handwritten character recognition

**`np-n/Devanagari-Handwriting-Recognition`**

https://github.com/np-n/Devanagari-Handwriting-Recognition

This is a Devanagari handwriting-recognition repository and can be useful for examining character-level classification and dataset handling.

### C. Modern historical-manuscript HTR engine

**Kraken**

https://github.com/mittagessen/kraken

Kraken is the most important architectural component to investigate for EyM. It is designed for historical and non-Latin material and supports trainable layout analysis, reading order and recognition, multiple scripts, and manuscript-oriented output. It is explicitly intended for historical documents rather than only modern printed OCR.

### D. Full manuscript workflow

**eScriptorium**

https://github.com/eScriptorium/eScriptorium

https://escriptorium.eu/

eScriptorium provides a complete web workflow for image ingestion, segmentation, transcription, correction and model training, using Kraken as its recognition engine. Its architecture is particularly relevant to EyM because EyM needs a scholarly correction loop rather than a one-shot OCR result.

### E. Devanagari post-OCR correction

**`tusharislampure29/sanskrit-ocr-correction`**

https://github.com/tusharislampure29/sanskrit-ocr-correction

This is exceptionally relevant to EyM. It implements a byte-level ByT5 correction model for noisy Sanskrit/Devanagari OCR and includes a linguistically grounded Devanagari OCR-noise engine. It explicitly models errors such as:

- vowel-sign confusion;
- vowel-sign deletion;
- anusvāra/chandrabindu confusion;
- visarga loss;
- halant/virāma errors;
- conjunct splitting;
- visually confusable consonants;
- danda errors;
- word-boundary errors;
- Unicode normalisation errors;
- numeral confusion.

EyM should borrow this **error-taxonomy concept**, even if the final model is trained on real manuscript data.

### F. Sanskrit post-OCR correction benchmark/data

**`ayushbits/pe-ocr-sanskrit`**

https://github.com/ayushbits/pe-ocr-sanskrit

This provides post-edited Sanskrit OCR data associated with the EMNLP 2022 benchmark *A Benchmark and Dataset for Post-OCR text correction in Sanskrit*.

### G. Devanagari grapheme dataset

**Himalaya AI – Devanagari OCR Graphemes**

https://huggingface.co/datasets/himalaya-ai/devanagari_ocr_graphemes

This provides grapheme-level Devanagari images paired with Unicode text. It contains approximately 57,000 training examples and is particularly useful for building the **canonical grapheme reference layer** proposed below.

### H. Large Indic handwriting dataset

**IIIT-INDIC-HW**

https://cvit.iiit.ac.in/ihtr2022/dataset.html

The 2022 Indic Handwritten Text Recognition competition includes a large Devanagari word-image corpus. The Devanagari training split contains 69,583 word images, with 12,708 validation images and 13,869 test images.

A Hugging Face conversion is also available:

https://huggingface.co/datasets/c3rl/IIIT-INDIC-HW-WORDS-Hindi

This is much more useful to EyM than isolated-character datasets because it exposes handwriting in word-level context.

---

# 3. The conceptual change required in EyM

Do **not** build:

> handwritten image → font classification → Unicode

Build:

> handwritten image → Devanagari grapheme interpretation → canonical Unicode

The distinction is crucial.

A scribe does not write a font.

A font is a typographic representation of an abstract character/glyph system.

A scribe produces a **hand**, whose forms are influenced by:

- script tradition;
- period;
- region;
- training;
- individual habit;
- writing instrument;
- writing surface;
- ink;
- incision technique;
- writing speed;
- fatigue;
- spacing;
- lineation;
- manuscript format;
- copying exemplar;
- abbreviations;
- scribal conventions;
- damage and degradation.

Therefore EyM should learn **style variation around canonical grapheme identity**, rather than trying to assign every manuscript hand to a commercial or typographic font.

---

# 4. The proposed EyM model

## 4.1 Canonical Devanagari Reference Layer

Construct a reference bank containing:

### Independent vowels

अ आ इ ई उ ऊ ऋ ॠ ऌ ए ऐ ओ औ

### Consonants

क ख ग घ ङ  
च छ ज झ ञ  
ट ठ ड ढ ण  
त थ द ध न  
प फ ब भ म  
य र ल व  
श ष स ह

### Signs and modifiers

ँ ं ः ् ॐ

and the full set of:

- vowel signs;
- virāma/halant;
- nukta forms;
- Vedic marks where required;
- digits;
- punctuation;
- danda;
- double danda;
- avagraha;
- spacing conventions.

But the unit of recognition should preferably be a **grapheme/akṣara cluster**, not merely an isolated Unicode code point.

For example:

क + ् + ष

should be represented internally as a Devanagari grapheme sequence corresponding to:

> क्ष

rather than forcing the visual model to recognise three independent shapes.

---

# 5. The "perfect font table" idea should be implemented as a prototype manifold

The user's proposed "perfect and most embellished Devanagari font table" should become a **canonical visual reference manifold**, not a single font.

Create a large synthetic glyph bank from many high-quality Devanagari fonts.

For each grapheme:

```text
grapheme: क्ष

font_001 → rendered form
font_002 → rendered form
font_003 → rendered form
...
font_N   → rendered form
```

Then add synthetic transformations:

- stroke-width variation;
- slant;
- scaling;
- compression;
- stretching;
- rotation;
- blur;
- erosion;
- dilation;
- broken shirorekha;
- uneven ink;
- missing matra;
- joined strokes;
- disconnected strokes;
- background texture;
- paper texture;
- palm-leaf texture;
- incision-like strokes.

The result becomes:

> **Canonical Devanagari Grapheme Manifold**

rather than:

> one ideal font.

This gives EyM a much stronger reference space.

---

# 6. Why multiple fonts are necessary

A single "perfect" Devanagari font would create a hidden bias.

Different fonts encode different typographic decisions.

The same abstract grapheme can have:

- different terminal shapes;
- different curvature;
- different proportions;
- different conjunct construction;
- different matra geometry;
- different shirorekha thickness;
- different stroke contrast.

Therefore EyM should learn the invariant:

> **identity of the grapheme**

while treating its visible shape as a variable.

Conceptually:

```text
Observed image
      │
      ▼
Visual encoder
      │
      ├── writing-medium features
      ├── degradation features
      ├── hand/style features
      └── glyph identity features
                  │
                  ▼
       Canonical grapheme space
                  │
                  ▼
          Unicode sequence
```

---

# 7. Writing-medium model

The manuscript medium must not be treated as cosmetic metadata.

The physical writing process can alter the visual form.

The EyM classifier should therefore estimate a `medium_profile`.

Suggested classes:

```text
paper_ink
paper_pencil
paper_pen
birch_bark
palm_leaf_incised
palm_leaf_ink
stone_inscription
metal_inscription
wood
cloth
mixed_or_unknown
```

Each profile should have its own augmentation and preprocessing strategy.

---

# 8. Why palm leaf is especially important

Historical evidence shows that palm-leaf writing can involve incision with a metal stylus and subsequent darkening of the grooves.

The physical process can produce a fundamentally different image from ink-on-paper writing.

The IIT Bombay documentation describes palm-leaf incision with a stylus and explains how black/charcoal mixtures can be applied after incision to make the grooves visible.

Source:

https://www.idc.iitb.ac.in/resources/dt-july-2009/Palm.pdf

The University of Michigan Museum of Art similarly documents Devanagari(?) palm-leaf material in which letters were incised and the incision darkened with charcoal dust.

https://umma.umich.edu/objects/palm-leaf-from-an-unidentified-manuscript-with-devanagiri-script-1997-2-41/

Therefore:

```text
ink stroke ≠ incised groove
```

and EyM should not force both through an identical image-normalisation pipeline.

---

# 9. Material-aware preprocessing

## Paper

Possible transformations:

- uneven illumination;
- ink bleed;
- faded ink;
- paper yellowing;
- foxing;
- stains;
- folds;
- page curvature;
- scanning noise;
- ink feathering.

## Palm leaf

Possible transformations:

- longitudinal fibres;
- curved surface;
- incision shadows;
- uneven blackening;
- abrasion;
- groove discontinuity;
- leaf cracks;
- holes;
- binding damage;
- darkened background;
- variable contrast.

## Stone

Possible transformations:

- chisel width;
- weathering;
- erosion;
- surface roughness;
- shadows;
- moss;
- cracks;
- uneven depth;
- oblique illumination.

The model should estimate these conditions before recognition.

---

# 10. Writer/style embedding

EyM should maintain a separate latent vector:

```text
style_embedding
```

This is not the text.

It represents how the scribe writes.

For example:

```text
Hand A
  ├── narrow glyphs
  ├── long shirorekha
  ├── compressed matras
  ├── high vertical strokes
  └── frequent conjunct compression

Hand B
  ├── broad glyphs
  ├── broken shirorekha
  ├── rounded terminals
  ├── large vowel signs
  └── wide spacing
```

The same glyph identity may therefore appear very differently in the two hands.

The model should learn:

```text
image = content + hand/style + medium + degradation
```

and learn to separate these factors.

---

# 11. Recommended model architecture

A practical EyM implementation should contain at least six stages.

```text
                         MANUSCRIPT IMAGE
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Document analyser   │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        Layout/region     Medium/material    Quality/degradation
          analysis           classifier          estimator
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    Line / word segmentation
                               │
                               ▼
                     Devanagari HTR encoder
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
           grapheme ID    style embedding  visual features
                 │             │             │
                 └─────────────┼─────────────┘
                               ▼
                  Canonicalisation decoder
                               │
                               ▼
                     N-best Unicode text
                               │
                               ▼
                Sanskrit / Indic language model
                               │
                               ▼
                     candidate re-ranking
                               │
                               ▼
                  confidence + alternatives
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
          accepted Unicode             human review
```

---

# 12. Do not force early correction

This is one of the most important design principles.

EyM must distinguish:

### Observation

"What does the image appear to contain?"

from:

### Interpretation

"What is the most probable reading?"

from:

### Normalisation

"How should that reading be represented in Unicode?"

and from:

### Scholarly correction

"What reading should an editor accept after considering the textual tradition?"

These are not the same operation.

Recommended internal representation:

```json
{
  "observed": "visual hypothesis",
  "recognition": "क...",
  "normalised": "क...",
  "linguistic_candidate": "क...",
  "confidence": 0.91,
  "alternatives": [
    {
      "text": "ख...",
      "confidence": 0.06
    }
  ],
  "requires_review": false
}
```

---

# 13. N-best recognition is essential

Instead of:

```text
image → एक answer
```

use:

```text
image
 ↓
candidate 1   0.71
candidate 2   0.19
candidate 3   0.07
candidate 4   0.03
```

Then apply contextual ranking.

Example:

```text
visual candidates:

धर्मस्य
धर्मस्स
धर्मस्यः
धर्मस्यं

contextual ranking:

धर्मस्य
```

The system should be able to explain that the decision came from:

- visual evidence;
- grapheme compatibility;
- lexical probability;
- Sanskrit morphology;
- local context.

It must never silently convert an uncertain visual observation into a supposedly certain scholarly fact.

---

# 14. Sanskrit-aware candidate ranking

For Sanskrit manuscripts, EyM should have a second linguistic layer.

Possible components:

1. Sanskrit lexicon;
2. morphological analyser;
3. sandhi-aware candidate generator;
4. known textual corpus;
5. n-gram language model;
6. transformer language model;
7. manuscript-specific vocabulary;
8. domain vocabulary;
9. parallel editions;
10. user-provided reference text.

The language model should rank candidates, not manufacture arbitrary text.

A useful scoring function is:

```text
TotalScore =
    α × VisualScore
  + β × GraphemeScore
  + γ × LanguageScore
  + δ × LexicalScore
  + ε × ManuscriptContextScore
  - ζ × UnsupportedCorrectionPenalty
```

The final term is important.

EyM should penalise corrections that have weak visual support.

---

# 15. A critical rule for scholarly OCR

Never allow:

> "The model thinks this is probably a familiar Sanskrit word"

to override:

> "The image actually supports a different reading."

The language model is a **reader-assistance mechanism**, not an authority over the manuscript.

---

# 16. Grapheme-level recognition

Devanagari should be represented at the grapheme/akṣara level.

Example:

```text
क
कि
की
कु
कू
के
कै
को
कौ
क्र
क्त
क्ष
ज्ञ
```

These should not all be treated as unrelated characters.

Represent:

```text
base consonant
+
vowel sign
+
virama
+
conjunct components
+
additional marks
```

Internally:

```text
क् + ष → क्ष
```

but the visual model may recognise the whole grapheme.

This helps with:

- conjuncts;
- matras;
- reordering;
- ligatures;
- damaged strokes;
- partial glyphs.

---

# 17. Unicode normalisation

All final Devanagari output should be normalised to Unicode NFC.

Do not compare visually equivalent strings without Unicode normalisation.

Store:

```text
raw_model_output
unicode_nfc_output
```

separately.

For scholarly preservation, it is also useful to store:

```text
diplomatic_transcription
normalised_transcription
```

as separate layers.

---

# 18. Diplomatic vs normalised transcription

EyM should eventually support two outputs.

### Diplomatic

Attempts to reproduce what the manuscript actually shows.

### Normalised

Represents the intended readable Devanagari text in canonical Unicode.

Example:

```text
MANUSCRIPT:
[image]

DIPLOMATIC:
धर्मस्य ...

NORMALISED:
धर्मस्य ...

EDITORIAL:
धर्मस्य ...
```

If the manuscript has an unusual spelling, EyM should not automatically erase it.

---

# 19. Reference-font rendering

After recognition, render the Unicode result in one or more high-quality Devanagari fonts.

This is **not** because the font is the source of truth.

It is because rendering gives the reader a clean canonical representation.

Recommended UI:

```text
MANUSCRIPT IMAGE

        ↓

EyM READING

धर्मक्षेत्रे कुरुक्षेत्रे

        ↓

CANONICAL RENDERING

धर्मक्षेत्रे कुरुक्षेत्रे

        ↓

ALTERNATIVES

धर्मक्षेत्रे कुरुक्षेत्रे
धर्मक्षेत्रे कुरुक्षेत्र

        ↓

CONFIDENCE

97.2%
```

---

# 20. Visual glyph reconstruction should be optional

A useful advanced feature is:

```text
Observed glyph
       ↓
Nearest canonical glyph
       ↓
Canonical rendering
```

The UI can show:

```text
[manuscript glyph] → [canonical Devanagari glyph]
```

This is valuable for human validation.

But the reconstructed glyph must never replace the original manuscript image.

---

# 21. Dataset strategy

EyM should combine several classes of data.

## Tier 1 — isolated graphemes

Purpose:

- establish canonical Devanagari visual space;
- learn grapheme morphology.

Use:

- Devanagari OCR Graphemes;
- DHCD;
- other clean Devanagari character datasets.

## Tier 2 — handwritten words

Purpose:

- learn handwriting variability;
- learn grapheme sequencing.

Use:

- IIIT-INDIC-HW;
- Hindi/Devanagari handwriting datasets.

## Tier 3 — handwritten lines/pages

Purpose:

- real HTR;
- spacing;
- line structure;
- word boundaries;
- contextual decoding.

Use:

- DevanagariHTR-derived material;
- eScriptorium/Kraken training data;
- institutional manuscript collections where licensing permits.

## Tier 4 — historical manuscripts

Purpose:

- actual EyM domain adaptation.

This must eventually become the most important dataset.

---

# 22. The EyM Manuscript Corpus

Create a project-specific corpus:

```text
EyM-Manuscript-Corpus/
│
├── paper/
│   ├── early/
│   ├── medieval/
│   ├── early_modern/
│   └── modern/
│
├── palm_leaf/
│   ├── ink/
│   └── incised/
│
├── birch_bark/
│
├── stone/
│
├── metal/
│
└── mixed/
```

Each image should have metadata.

Example:

```json
{
  "manuscript_id": "EYM-MSS-0001",
  "folio": "12r",
  "script": "Devanagari",
  "language": "Sanskrit",
  "medium": "paper",
  "writing_method": "ink",
  "estimated_period": "18th_century",
  "region": "unknown",
  "scribe": "unknown",
  "source": "institutional_collection",
  "image_license": "CC-BY",
  "ground_truth_status": "expert_verified"
}
```

---

# 23. Ground truth must be expert-grade

For a scholarly OCR system, ground truth should not be generated by another OCR engine and then treated as truth.

Recommended hierarchy:

```text
Level 0:
unverified OCR

Level 1:
machine-generated transcription

Level 2:
human corrected transcription

Level 3:
trained Devanagari reader verified

Level 4:
Sanskrit/Indic manuscript specialist verified

Level 5:
critical-edition / diplomatic transcription
```

Training should preferably prioritise Levels 3–5.

---

# 24. Active learning

EyM should learn from its own correction interface.

Whenever a scholar changes:

```text
OCR output
```

to:

```text
accepted reading
```

store the pair:

```text
image crop
+
model prediction
+
human correction
```

Then periodically retrain/fine-tune.

This creates:

> **EyM manuscript-specific learning**

A model that becomes better at the actual corpus being digitised.

---

# 25. Writer-specific adaptation

When enough pages from one manuscript are available, create a temporary writer/manuscript profile.

Example:

```text
GLOBAL MODEL
      ↓
Manuscript adaptation
      ↓
EYM-MSS-023
      ↓
specific hand
      ↓
better recognition
```

This is often more realistic than demanding a universal model that recognises every possible hand equally well.

---

# 26. Two-pass recognition

Recommended:

### Pass 1 — universal

Recognise using the general Devanagari model.

### Pass 2 — manuscript-adapted

Use:

- known hand;
- known medium;
- known script;
- known vocabulary;
- corrections from previous pages.

This should substantially improve consistency across a manuscript.

---

# 27. Page-to-page consistency

If a manuscript contains 200 pages, EyM should not process each page as though it belongs to a completely new world.

It should maintain:

```text
manuscript profile
```

containing:

- writer embedding;
- glyph-shape statistics;
- common abbreviations;
- punctuation habits;
- preferred conjunct forms;
- unusual letter forms;
- recurring OCR confusions;
- vocabulary;
- line spacing;
- page geometry.

This is one of the strongest opportunities for improving accuracy.

---

# 28. Error memory

For every manuscript, maintain:

```json
{
  "manuscript_id": "EYM-MSS-023",
  "confusions": [
    ["व", "ब"],
    ["श", "ष"],
    ["ि", "ी"],
    ["ं", "ँ"]
  ]
}
```

But do not hard-code the corrections.

Store them as probabilities:

```json
{
  "visual_confusion": {
    "व→ब": 0.18,
    "श→ष": 0.31,
    "ि→ी": 0.12
  }
}
```

---

# 29. Training augmentation

Build a Devanagari-specific augmentation engine.

## Geometric

- scale;
- shear;
- rotation;
- perspective;
- elastic deformation;
- local warping.

## Stroke

- dilation;
- erosion;
- broken stroke;
- stroke thinning;
- stroke thickening;
- partial erasure;
- joining.

## Shirorekha

- broken headline;
- uneven headline;
- merged headline;
- missing segment;
- excessive thickness.

## Matra

- displaced matra;
- missing matra;
- merged matra;
- shortened matra;
- stretched matra.

## Material

- paper;
- palm leaf;
- stone;
- wood;
- birch bark.

## Capture

- blur;
- low DPI;
- uneven illumination;
- camera perspective;
- compression;
- shadows.

---

# 30. Synthetic handwriting is useful but insufficient

Rendered fonts can provide enormous training data.

But:

```text
font rendering ≠ handwriting
```

Therefore use synthetic data for:

- grapheme coverage;
- rare conjuncts;
- Vedic marks;
- unusual combinations;
- controlled corruption.

Use real handwriting for:

- actual writer variation;
- stroke order effects;
- spacing;
- ligature variation;
- material effects.

---

# 31. The ideal training mixture

A reasonable starting target:

```text
20% clean synthetic canonical glyph/grapheme data
20% synthetic distorted glyph/grapheme data
25% real handwritten word data
20% real handwritten line/page data
15% historical manuscript data
```

These percentages are starting hypotheses, not fixed requirements.

As the EyM corpus grows, the proportion of real manuscript data should increase.

---

# 32. Model families to investigate

## Kraken

Best candidate for the first manuscript-oriented HTR infrastructure.

Use it for:

- segmentation;
- line recognition;
- trainable recognition;
- historical documents.

## TrOCR

Useful for experimental end-to-end line/word recognition and fine-tuning.

Existing Devanagari and Sanskrit models exist, but their limitations must be examined carefully.

For example:

`Piyush3142/trocr-sanskrit-ocr`

https://huggingface.co/Piyush3142/trocr-sanskrit-ocr

is trained on Sanskrit manuscript images but reports relatively high CER/WER and is explicitly limited by its training domain.

`paudelanil/trocr-devanagari-2`

https://huggingface.co/paudelanil/trocr-devanagari-2

is another Devanagari handwritten-text model worth benchmarking.

## ByT5

Use primarily for post-OCR correction and candidate ranking.

It is particularly appropriate for noisy Devanagari because byte-level processing avoids unknown-character problems and handles arbitrary Unicode output.

---

# 33. Recommended first EyM architecture

Do not immediately replace the entire current OCR system.

Create:

```text
EyM OCR v2
│
├── Existing OCR
│
├── Devanagari HTR engine
│
├── Canonical Grapheme Normaliser
│
├── Contextual Candidate Ranker
│
├── Sanskrit OCR Corrector
│
├── Manuscript Profile
│
└── Human Review UI
```

This allows A/B comparison.

---

# 34. Candidate architecture using Kraken

```text
Image
 ↓
Kraken segmentation
 ↓
line crops
 ↓
EyM Devanagari recognition model
 ↓
N-best Unicode candidates
 ↓
grapheme normalisation
 ↓
ByT5 / Sanskrit correction model
 ↓
lexical + linguistic ranking
 ↓
confidence
 ↓
Unicode NFC
 ↓
HTML / database / downloadable text
```

---

# 35. Why Kraken is particularly suitable

Kraken explicitly targets historical and non-Latin material and provides trainable layout analysis, reading order and recognition.

Its documentation makes an important point:

> a recognition model learns to read a specific script/typeface or scribal tradition, and a model trained on the wrong document family may fail.

This aligns closely with the EyM concept.

A universal Devanagari recogniser should therefore be treated as a base model, followed by manuscript-specific adaptation.

---

# 36. eScriptorium as a research reference

Even if EyM does not use eScriptorium directly, study its workflow.

Its architecture provides:

```text
image ingestion
→ segmentation
→ transcription
→ manual correction
→ model training
→ export
```

That is exactly the scholarly loop EyM requires.

EyM should eventually provide an API comparable in spirit to:

```text
POST /ocr
POST /segment
POST /transcription
POST /correct
POST /feedback
POST /train
GET  /manuscript/{id}/profile
```

---

# 37. Confidence model

Every output should have at least three confidence values:

```text
visual_confidence
language_confidence
final_confidence
```

Example:

```text
Visual:       0.82
Language:     0.96
Final:        0.88
```

A second useful metric:

```text
visual_language_disagreement
```

Example:

```text
Visual model:   धर्मस्य
Language model: धर्मस्य

Agreement: high
```

versus:

```text
Visual model:   धर्मस्स
Language model: धर्मस्य

Agreement: low
Review: required
```

---

# 38. Human-review threshold

Suggested initial policy:

```text
confidence >= 0.97
    auto-accept

0.85–0.97
    accept but mark as machine-normalised

0.60–0.85
    show alternatives

< 0.60
    mandatory human review
```

These are starting thresholds and must be calibrated against a held-out expert-verified dataset.

---

# 39. Never hide uncertainty

The user interface should visually distinguish:

```text
CERTAIN
PROBABLE
UNCERTAIN
EDITORIAL
```

The scholar should be able to click an uncertain grapheme and see:

```text
Top candidate:
ष  0.61

Alternative:
श  0.27

Alternative:
स  0.12
```

---

# 40. Evaluation

Do not evaluate EyM only with character accuracy.

Use:

### CER

Character Error Rate.

### WER

Word Error Rate.

### Grapheme Error Rate

More meaningful for Devanagari than raw Unicode code-point error alone.

### Exact-line accuracy

Useful for finished transcription.

### Confidence calibration

Does 90% confidence actually mean approximately 90% reliability?

### Human correction rate

What proportion of lines require intervention?

### Scholarly acceptance rate

What proportion of automatically produced readings are accepted by experts?

---

# 41. Error taxonomy

Create a permanent EyM error taxonomy.

```text
E01  consonant confusion
E02  vowel confusion
E03  matra deletion
E04  matra substitution
E05  matra displacement
E06  anusvara/chandrabindu
E07  visarga
E08  virama/halant
E09  conjunct decomposition
E10  conjunct substitution
E11  shirorekha segmentation
E12  word boundary
E13  punctuation/danda
E14  numeral
E15  Unicode normalisation
E16  damage-induced ambiguity
E17  material-induced distortion
E18  writer-style ambiguity
E19  abbreviation
E20  genuine scribal variant
```

This taxonomy should be part of every evaluation report.

---

# 42. A critical distinction: OCR error vs scribal variant

EyM must not assume every deviation from modern Unicode Devanagari is an OCR error.

Possible cases:

```text
image
 ↓
unusual manuscript form
```

could mean:

1. OCR mistake;
2. scribal abbreviation;
3. orthographic variant;
4. historical glyph form;
5. ligature;
6. damaged glyph;
7. genuine textual variant.

The system must preserve this distinction.

---

# 43. Manuscript profile should include palaeographic information

Where scholarly metadata exists, include:

```text
script family
script style
region
date/period
scribe
material
instrument
ink
incision method
language
genre
manuscript tradition
```

Do not infer historical date or region merely from a machine vision embedding.

If the system predicts palaeographic similarity, label it:

> **model-derived similarity**

not as an established historical fact.

---

# 44. Scribe mental state

The user's conceptual model includes the mental and physical condition of the scribe.

This should be treated carefully.

A model cannot reliably infer:

> "the scribe was tired"

from a page.

It can, however, detect measurable proxies such as:

- increasing character size;
- baseline drift;
- increasing irregularity;
- stroke pressure variation;
- spacing changes;
- omissions;
- corrections;
- overwriting;
- tremor-like distortion.

Therefore EyM should store:

```text
observed_variation_features
```

rather than:

```text
mental_state = tired
```

Any psychological interpretation must remain scholarly speculation unless independently documented.

---

# 45. Context from previous and subsequent lines

A major improvement should be **document-level decoding**.

Instead of:

```text
line 17 → independently recognised
```

use:

```text
line 15
line 16
line 17
line 18
line 19
      ↓
document-context decoder
```

The model can learn:

- recurring words;
- recurring ligatures;
- recurring abbreviations;
- scribe-specific shapes;
- section vocabulary;
- metre;
- formulaic expressions.

---

# 46. Sanskrit metre as a possible advanced constraint

For metrical Sanskrit manuscripts, metre can provide a powerful secondary validation layer.

If the text appears to be a śloka, EyM can calculate whether candidate readings satisfy expected syllabic structure.

But metre must never override manuscript evidence.

Use it as:

```text
candidate-ranking signal
```

not:

```text
automatic correction authority
```

---

# 47. Known-text matching

Where the manuscript is known or suspected to contain a known work, EyM may compare candidates against:

- public-domain editions;
- Sanskrit corpora;
- dictionaries;
- parallel manuscripts;
- catalogued editions.

But the UI must clearly label:

```text
image-derived reading
```

and:

```text
parallel-text suggestion
```

These are not interchangeable.

---

# 48. Proposed EyM output schema

```json
{
  "manuscript_id": "EYM-MSS-0001",
  "page": 12,
  "line": 7,
  "image_region": [x, y, width, height],
  "medium": {
    "class": "paper_ink",
    "confidence": 0.94
  },
  "script": {
    "class": "Devanagari",
    "confidence": 0.99
  },
  "observed_text": "धर्मस्स",
  "normalised_text": "धर्मस्य",
  "confidence": 0.91,
  "alternatives": [
    {
      "text": "धर्मस्स",
      "score": 0.17
    }
  ],
  "corrections": [
    {
      "from": "स्स",
      "to": "स्य",
      "reason": "visual+linguistic",
      "confidence": 0.91
    }
  ],
  "review_required": false
}
```

---

# 49. Database model

Suggested tables:

```text
manuscripts
folios
regions
lines
glyph_observations
graphemes
transcriptions
candidates
corrections
writers
writer_embeddings
medium_profiles
models
model_versions
human_reviews
training_examples
```

This makes the system auditable.

---

# 50. Model versioning

Never overwrite a production OCR model without retaining its version.

Example:

```text
eym-devanagari-htr-v1
eym-devanagari-htr-v2
eym-devanagari-htr-v2.1
eym-sanskrit-corrector-v1
```

Every transcription should store:

```text
recognition_model_version
correction_model_version
normalisation_version
```

---

# 51. Reproducibility

For scholarly use, every automatically generated text should be reproducible from:

```text
image
+
model versions
+
configuration
+
metadata
```

This is essential if EyM is used for academic publication.

---

# 52. Recommended GitHub integration order

## Phase 1

Study and benchmark:

1. `mittagessen/kraken`
2. `subhrajyotidasgupta/DevanagariHTR`
3. `np-n/Devanagari-Handwriting-Recognition`
4. IIIT-INDIC-HW
5. Devanagari OCR Graphemes

## Phase 2

Add correction:

6. `tusharislampure29/sanskrit-ocr-correction`
7. `ayushbits/pe-ocr-sanskrit`

## Phase 3

Build canonicalisation:

8. multi-font Devanagari grapheme renderer
9. grapheme prototype bank
10. synthetic handwriting generator

## Phase 4

Train on EyM manuscript corpus.

---

# 53. Recommended first prototype

Do not attempt all scripts and all materials at once.

Start with:

```text
Sanskrit
+
Devanagari
+
ink on paper
```

Then:

```text
Sanskrit
+
Devanagari
+
palm-leaf ink/incision
```

Then:

```text
historical/degraded paper
```

Then:

```text
stone/inscription
```

This makes error analysis possible.

---

# 54. Suggested experimental baseline

Build four systems and compare them.

### Baseline A

Current EyM OCR.

### Baseline B

Devanagari HTR model.

### Baseline C

HTR + canonical grapheme normalisation.

### Baseline D

HTR + canonicalisation + Sanskrit post-correction + manuscript adaptation.

The goal is to demonstrate incremental improvement.

---

# 55. Expected result

The desired transformation is:

```text
RAW MANUSCRIPT
       ↓
VISUAL RECOGNITION
       ↓
CANONICAL GRAPHEME INTERPRETATION
       ↓
CONTEXTUAL CANDIDATE GENERATION
       ↓
LINGUISTIC RANKING
       ↓
UNICODE NORMALISATION
       ↓
SCHOLARLY REVIEW
```

not simply:

```text
RAW IMAGE
 ↓
OCR
 ↓
TEXT
```

---

# 56. Research principle

The central research hypothesis for EyM should be formulated as:

> **Devanagari manuscript recognition can be improved by modelling handwriting variation as a transformation of canonical graphemic forms, while jointly accounting for writer-specific style, writing medium, document degradation and linguistic context.**

This is a much stronger research proposition than:

> "recognise handwritten Devanagari characters."

---

# 57. Suggested name for the component

Avoid calling it "Font Detector".

Recommended names:

### EyM Devanagari Hand Normaliser

or:

### EyM Canonical Devanagari Interpreter

or:

### EyM Scribal Devanagari HTR

The most technically accurate internal name would be:

> **Devanagari Grapheme Canonicalisation and Contextual HTR**

---

# 58. Research terminology

Use these terms consistently:

| Concept | Preferred term |
|---|---|
| perfect font | canonical grapheme reference |
| handwriting style | writer/hand style |
| font guessing | glyph-form canonicalisation |
| OCR correction | post-recognition correction |
| manuscript distortion | material/writing-induced variation |
| correct text | candidate reading |
| final corrected text | accepted scholarly transcription |
| font matching | canonical-form similarity |
| character | grapheme/akṣara where appropriate |

---

# 59. Recommended source literature and knowledge base

## Historical writing materials

IIT Bombay / IDC documentation on palm-leaf manuscripts:

https://www.idc.iitb.ac.in/resources/dt-july-2009/Palm.pdf

This documents stylus-based incision, writing mechanics, correction difficulty and material effects.

University of Michigan Museum of Art:

https://umma.umich.edu/objects/palm-leaf-from-an-unidentified-manuscript-with-devanagiri-script-1997-2-41/

Useful material evidence for Devanagari(?) palm-leaf writing.

National Mission for Manuscripts:

https://www.namami.gov.in/cataloguing-manuscripts

Useful for understanding the scribe, manuscript metadata and cataloguing conventions.

University of Chicago Library:

https://www.lib.uchicago.edu/collex/exhibits/envisioning-south-asia-texts-scholarship-legacies/glimpses-past/

Useful contextual material on South Asian manuscript culture and writing media.

## HTR / OCR

Kraken:

https://github.com/mittagessen/kraken

eScriptorium:

https://escriptorium.eu/

IIIT-INDIC-HW:

https://cvit.iiit.ac.in/ihtr2022/dataset.html

HTR-United:

https://htr-united.github.io/catalog.html

The HTR-United catalogue is useful for discovering additional ground-truth collections.

## Devanagari datasets

Devanagari OCR Graphemes:

https://huggingface.co/datasets/himalaya-ai/devanagari_ocr_graphemes

IIIT-INDIC-HW Hindi:

https://huggingface.co/datasets/c3rl/IIIT-INDIC-HW-WORDS-Hindi

## Sanskrit OCR correction

Sanskrit OCR correction:

https://github.com/tusharislampure29/sanskrit-ocr-correction

Sanskrit post-OCR correction benchmark:

https://github.com/ayushbits/pe-ocr-sanskrit

## South Asian manuscript HTR research

Oxford's Digital Humanities & Hindu Studies project:

https://www.theology.ox.ac.uk/digital-humanities-hindu-studies-creating-ai-models-handwriting-and-text-recognition-south-asian

This project is particularly important because it explicitly works toward Devanagari OCR for ancient Sanskrit manuscripts and includes palm-leaf manuscripts from South Asian traditions.

---

# 60. Oxford project: particularly important for EyM

The Oxford project should be treated as a major research lead.

It reports work on Devanagari OCR for ancient Sanskrit manuscripts using Transkribus AI and includes palm-leaf manuscripts from Śākta and Vaiṣṇava traditions.

This demonstrates that the exact research problem envisioned for EyM is already being pursued in the academic manuscript-digitisation community.

EyM should therefore benchmark against, learn from and potentially collaborate with this research ecosystem rather than building an isolated OCR experiment.

---

# 61. Important limitation of existing Devanagari datasets

The standard DHCD-style character datasets are valuable, but they are not sufficient for EyM.

They generally contain:

```text
isolated characters
+
clean images
+
fixed classes
```

EyM requires:

```text
whole pages
+
real writers
+
historical hands
+
material variation
+
degradation
+
grapheme sequences
+
context
```

Therefore the existing datasets should be considered **pretraining resources**, not the final corpus.

---

# 62. Synthetic canonical font bank

Create an EyM-generated dataset:

```text
eym_devanagari_canonical/
│
├── fonts/
│   ├── font001/
│   ├── font002/
│   └── ...
│
├── graphemes/
│   ├── independent_vowels/
│   ├── consonants/
│   ├── matras/
│   ├── conjuncts/
│   ├── numerals/
│   └── punctuation/
│
└── augmented/
    ├── paper/
    ├── palm_leaf/
    ├── stone/
    └── degraded/
```

Each image must retain its Unicode label.

---

# 63. Synthetic-to-real strategy

Train:

```text
canonical synthetic
       ↓
synthetically distorted
       ↓
real handwriting
       ↓
historical manuscript
```

Use curriculum learning.

Do not train only on synthetic data and assume historical generalisation.

---

# 64. Nearest canonical glyph

For each visual observation:

```text
observed glyph embedding
        ↓
nearest canonical prototypes
        ↓
candidate graphemes
```

For example:

```text
Observed:
[distorted handwritten form]

Nearest canonical candidates:
क्ष  0.63
क्श  0.21
क्ष्  0.09
क  0.07
```

Then contextual decoding can resolve the candidate.

---

# 65. Contrastive learning

A particularly suitable research direction is contrastive learning.

Positive pairs:

```text
same grapheme
different font
different writer
different medium
```

Negative pairs:

```text
different grapheme
```

Training objective:

> pull all visual realisations of the same grapheme together while separating different graphemes.

This directly implements the user's idea of recognising the underlying "perfect" Devanagari form beneath handwriting variation.

---

# 66. Two embeddings

The recognition network should ideally learn:

```text
content_embedding
style_embedding
```

For the same word written by two scribes:

```text
content embedding ≈ same
style embedding ≠ same
```

For two different words written by the same scribe:

```text
content embedding ≠ same
style embedding ≈ same
```

This disentanglement is a strong research direction for EyM.

---

# 67. Do not overfit to modern Devanagari typography

Modern Unicode fonts are a reference representation.

They are not historical truth.

A manuscript may legitimately contain forms that differ from modern typography.

Therefore the canonical layer should include:

```text
modern Unicode form
+
historical glyph variants
+
scribal variants
```

where evidence exists.

---

# 68. Versioned palaeographic reference bank

Eventually create:

```text
EYM-PALAEOGRAPHY/
│
├── early_nagari/
├── nagari/
├── devanagari/
├── regional_variants/
└── modern_devanagari/
```

This should be curated by specialists rather than inferred solely from fonts.

---

# 69. Scholarly review interface

The interface should allow:

```text
[image crop]

Machine:
धर्मस्स

Canonical:
धर्मस्य

Alternative:
धर्मस्स

Suggested because:
- visual similarity
- Sanskrit lexical probability
- manuscript profile

[Accept] [Reject] [Choose alternative] [Edit]
```

The human decision becomes training data.

---

# 70. Audit trail

Every correction should store:

```text
who
when
original OCR
machine suggestion
accepted reading
reason
model version
```

This makes EyM suitable for academic use.

---

# 71. Minimum viable implementation

For the first implementation, build only:

```text
1. Kraken/equivalent line segmentation
2. Devanagari HTR model
3. Canonical grapheme bank
4. N-best decoder
5. Sanskrit post-OCR corrector
6. Unicode NFC normalisation
7. confidence score
8. human correction interface
```

Do not start with palaeographic dating or psychological inference.

---

# 72. Recommended first experiment

Take 100 pages of real Devanagari Sanskrit manuscripts.

Split:

```text
60 pages training
20 pages validation
20 pages blind test
```

Annotate at line level.

Compare:

```text
A. existing EyM
B. Devanagari HTR
C. HTR + canonicalisation
D. HTR + canonicalisation + language model
E. HTR + canonicalisation + language model + manuscript adaptation
```

Report:

```text
CER
WER
grapheme error rate
exact-line accuracy
human correction rate
confidence calibration
```

---

# 73. Success criterion

EyM should not claim:

> "100% accurate OCR."

Instead:

> "EyM produces a ranked, auditable reading of Devanagari manuscript text, with canonical Unicode normalisation, explicit uncertainty, manuscript-specific adaptation and scholarly correction."

That is both technically credible and academically defensible.

---

# 74. Final implementation principle

The central insight of this skill is:

> **A manuscript is not a font rendered badly.**

It is the visible result of a writing act performed by a trained human using a particular script tradition, tool and material, under particular physical conditions.

Therefore the OCR system must learn the transformation:

```text
SCRIPT KNOWLEDGE
       +
SCRIBE HABIT
       +
WRITING INSTRUMENT
       +
WRITING MATERIAL
       +
PHYSICAL DEGRADATION
       +
GRAPHEMIC STRUCTURE
       +
LINGUISTIC CONTEXT
       ↓
OBSERVED MANUSCRIPT FORM
```

and perform the inverse operation:

```text
OBSERVED MANUSCRIPT FORM
       ↓
probable graphemic structure
       ↓
canonical Devanagari
       ↓
Unicode
       ↓
contextually ranked reading
       ↓
scholarly verification
```

This is the appropriate conceptual foundation for the next generation of EyM.

---

# 75. Immediate action plan for EyM

### Step 1
Benchmark the current EyM OCR on a fixed Devanagari manuscript test set.

### Step 2
Clone and evaluate:

```bash
git clone https://github.com/mittagessen/kraken
git clone https://github.com/subhrajyotidasgupta/DevanagariHTR
git clone https://github.com/np-n/Devanagari-Handwriting-Recognition
git clone https://github.com/tusharislampure29/sanskrit-ocr-correction
git clone https://github.com/ayushbits/pe-ocr-sanskrit
```

### Step 3
Download/prepare:

- IIIT-INDIC-HW;
- Devanagari grapheme dataset;
- DHCD;
- real EyM manuscript pages.

### Step 4
Create the canonical Devanagari grapheme bank.

### Step 5
Create the first Devanagari writer/style embedding model.

### Step 6
Train/fine-tune a Devanagari HTR model.

### Step 7
Add N-best decoding.

### Step 8
Add Sanskrit post-OCR correction.

### Step 9
Add manuscript-specific adaptation.

### Step 10
Add scholarly human review and feedback capture.

### Step 11
Evaluate against the blind expert-verified corpus.

### Step 12
Only then deploy the improved model to Brahmavidya.co.in.

---

# 76. Recommended repository structure for EyM

```text
eym/
├── ocr/
│   ├── segmentation/
│   ├── devanagari_htr/
│   ├── grapheme_encoder/
│   └── decoder/
│
├── canonical/
│   ├── fonts/
│   ├── graphemes/
│   ├── variants/
│   └── renderer/
│
├── handwriting/
│   ├── writer_encoder/
│   ├── style_embeddings/
│   └── adaptation/
│
├── materials/
│   ├── paper/
│   ├── palm_leaf/
│   ├── stone/
│   └── birch_bark/
│
├── language/
│   ├── sanskrit/
│   ├── lexicon/
│   ├── morphology/
│   ├── metre/
│   └── post_correction/
│
├── manuscripts/
│   ├── metadata/
│   ├── profiles/
│   └── ground_truth/
│
├── review/
│   ├── candidates/
│   ├── corrections/
│   └── audit/
│
├── evaluation/
│   ├── cer/
│   ├── wer/
│   ├── grapheme_error/
│   └── confidence/
│
└── docs/
    └── eym-devanagari-manuscript-htr.md
```

---

# 77. Key references

1. Kraken OCR/HTR  
   https://github.com/mittagessen/kraken

2. eScriptorium  
   https://escriptorium.eu/

3. DevanagariHTR  
   https://github.com/subhrajyotidasgupta/DevanagariHTR

4. Devanagari Handwriting Recognition  
   https://github.com/np-n/Devanagari-Handwriting-Recognition

5. IIIT-INDIC-HW  
   https://cvit.iiit.ac.in/ihtr2022/dataset.html

6. Devanagari OCR Graphemes  
   https://huggingface.co/datasets/himalaya-ai/devanagari_ocr_graphemes

7. Sanskrit OCR correction  
   https://github.com/tusharislampure29/sanskrit-ocr-correction

8. Sanskrit post-OCR benchmark  
   https://github.com/ayushbits/pe-ocr-sanskrit

9. Oxford South Asian Manuscript HTR project  
   https://www.theology.ox.ac.uk/digital-humanities-hindu-studies-creating-ai-models-handwriting-and-text-recognition-south-asian

10. IIT Bombay palm-leaf writing documentation  
    https://www.idc.iitb.ac.in/resources/dt-july-2009/Palm.pdf

11. National Mission for Manuscripts  
    https://www.namami.gov.in/cataloguing-manuscripts

12. HTR-United catalogue  
    https://htr-united.github.io/catalog.html

---

# 78. One-sentence specification

> **EyM shall recognise Devanagari manuscript writing not merely as distorted glyph images, but as variable manifestations of an underlying graphemic system, using canonical glyph prototypes, writer/style embeddings, material-aware visual modelling, contextual language constraints, N-best candidate generation, Sanskrit-aware post-OCR correction, Unicode normalisation and scholarly human verification to produce an auditable digital Devanagari transcription.**
