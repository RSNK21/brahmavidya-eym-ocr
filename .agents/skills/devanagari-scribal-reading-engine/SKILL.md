

## name: devanagari-scribal-reading-engine description: Use when designing, extending, or debugging OCR/HTR for handwritten, palm-leaf, inscriptional, or manuscript Devanāgarī (or sister scripts like Nandinagari, Modi, Sharada) — including scribal-variation modelling, material-conditioned degradation, akṣara lattice decoding, post-OCR correction, and Unicode rendering of predicted readings.

# Devanāgarī Scribal Reading Engine

Design and improve systems that read handwritten, etched, or carved Devanāgarī and emit correct Devanāgarī Unicode, by modelling the reader's inference rather than doing nearest-shape glyph extraction.

## Core principle

Never treat this as classification. Treat it as decoding over a noisy channel:

```
reading* = argmax_T  P(T) · P(observation | T, style, hand, material, instrument, condition)
```

Conditioning variables are estimated per manuscript, never assumed. Every accuracy improvement comes from making one of the four knowledge layers below explicit and inspectable.

## The four layers (always structure work this way)

1. **Layer 0 — canonical inventory.** Machine-readable akṣara table: 33 consonants + nukta forms, independent vowels, all dependent vowel signs (note pre-base ि, two-part ो ौ, below-base ु ू ृ), anusvāra/candrabindu/visarga/avagraha, virāma, repha and rakāra, eyelash ra, khaṇḍa ta, frequent and long conjuncts, digits, daṇḍa/double daṇḍa, Vedic marks (U+0951–0952, U+1CD0–1CFF). Each entry: canonical Unicode sequence, stroke-order skeleton, topological signature (stem position, śirorekhā coverage, loops, descender), confusables.

2. **Layer 1 — allograph/style atlas.** Named styles (regional, sectarian, chronological) each described along axes: śirorekhā treatment, stem length/slant, roundness vs angularity, loop closure, mātrā placement (including pṛṣṭhamātrā — e/ai written above the *preceding* akṣara), conjunct strategy (stacking vs half-forms), gemination-after-repha habit, nasal habit, numeral forms, decoration, line density. Every allograph maps back to its canonical Layer 0 unit plus a described deviation and, where possible, an executable decoder\_rule. A "hand" is an individual's realisation, learned per manuscript at run time; styles are stored data.

3. **Layer 2 — material and instrument physics.** Surface × tool × ink × condition profiles with both deformation geometry and degradation simulator parameters. Palm leaf + stylus + soot: long verticals suppressed (they split the leaf along its fibres), rounder and wider letters, light/intermittent śirorekhā, fibre striations, string holes, worm holes (frequently misread as anusvāra), fungal stain. Paper + reed pen: nib-angle stroke modulation, continuous śirorekhā across words, bleed-through from verso, foxing, later overwriting in red. Stone/copper + chisel: maximal angularity, uniform wide strokes, wider spacing, erosion and lichen, contrast that is geometric not tonal (use relief/RTI, never binarise first). Modern pen on ruled paper: uniform width, phone-photo blur and glare. Add per-line scribal-effort covariates (stroke-width trend, compression ratio vs line position, slant variance) and widen the decoder beam at line and folio ends.

4. **Layer 3 — the reader's linguistic prior.** Lexicon with inflected forms, morphology, sandhi, tradition-specific orthographic conventions, metre, corpus LM.

## Non-negotiable engineering rules

- **Akṣara, not codepoint, is the unit.** Report Akṣara Error Rate as the headline metric; add mātrā-only ER, conjunct ER (2 vs 3+ stacks), WER pre- and post-sandhi-splitting, lattice oracle recall@k, expected calibration error, and expert-minutes-per-folio. Always stratify by style and material. CER alone flatters Devanāgarī systems.

- **Recognisers emit lattices, not strings.** Per-slot ranked candidates with log-probs, including "mātrā absent", "this mark is damage not ink", and "these two slots are one stacked conjunct". Committing to a string before the linguistic layer discards the evidence needed for the reader's guess.

- **Decode as a cascade** with per-source attributed costs: lattice ∘ confusion FST (style/material-conditioned) ∘ orthographic-variant FST ∘ lexicon FST ∘ LM, then metre/formula re-ranking, then byte-level neural post-correction. Attribution per transducer is what lets the system explain each reading.

- **Reorder pre-base ि into logical order** and canonicalise half-form vs virāma, ZWJ for eyelash ra and khaṇḍa ta, NFC. Validate every emitted string with a grapheme-cluster validator that rejects impossible sequences.

- **Sanskrit word segmentation is a linguistic decision, not preprocessing.** Do not derive word boundaries from visual gaps; keep the unsegmented diplomatic string as primary. Metre is a strong cheap signal for resolving vowel-length and anusvāra ambiguity in verse — apply as a soft, tunable feature since manuscripts do contain unmetrical readings.

- **Dual output, always.** Diplomatic reading (never overwritten) plus normalised reading (reversible, rule-attributed), with per-akṣara confidence, ranked alternatives, reason strings, material/style/hand provenance, and needs\_human flags. Use critical-edition conventions for illegible and restored text. Never let a model invent plausible text over damage — that is the characteristic failure of large multimodal models here and is worse than a gap.

- **Per-manuscript adaptation gives the largest single jump.** Pseudo-label high-confidence lines, fine-tune a lightweight adapter for that hand, optionally synthesise handwriting in that hand from ~20 exemplars, and build a manuscript-local lexicon. This mirrors how a human reader warms up over the first folios.

- **Human corrections are the asset.** Log every keystroke from the correction editor as a training pair and a confusion-weight update; prioritise pages by uncertainty (active learning).

- **Do not conflate sister scripts with styles.** Nandināgarī, Modī, Newa/Rañjanā and Śāradā get their own inventories plus a transliteration bridge to Devanāgarī output.

## Cheapest wins first (recommend in this order for an existing OCR)

1. Post-correction layer over existing output using confusion graph + orthographic-variant grammar + lexicon + n-gram LM. No retraining.

2. Unicode hygiene/validator on the output path (fixes readings that are correct but render wrongly).

3. Material/style tagging at upload, routing to matched confusion weights and LM.

4. Expose posteriors and switch to lattice decoding.

5. Correction editor + feedback endpoint (the flywheel).

## Confusion sets to seed (replace weights with measured values)

Consonants: ब/व, भ/म, घ/ध, ख/रव, प/य, य/ष, थ/य, ट/ठ, ड/ङ, श/स, ल/ळ, ऋ/रु, त्र/ब. Mātrās: ि/ी, ु/ू, ु/ृ, े/ै, ो/ौ, ं/ँ, ं/absent. Damage-vs-diacritic: anusvāra vs worm hole, visarga vs double puncture, ऽ vs ३. Structural: half-form+base vs consonant+virāma, vertical stack vs two akṣaras, । vs १, ॥ vs ११.

## Orthographic variants that are conventions, not errors

Gemination after repha (धर्म ⇄ धर्म्म), anusvāra for class nasal (संस्कृत ⇄ सन्स्कृत), avagraha omission, vowel-length laxity (high cost — often a genuine variant of scholarly interest), sibilant laxity श/ष/स (condition cost on style), ब/व interchange, doubling before semivowels, visarga before sibilants, pṛṣṭhamātrā reordering (zero cost when the style flag is set), scriptio continua.

## Prior art to build on rather than reinvent

AnciDev (ancient Devanāgarī manuscript HTR dataset), IIIT-HW-Dev and CVIT Indic handwriting work, DHCD (saturated character benchmark), DohaScript (multi-writer continuous Hindi); Indiscapes/Indiscapes2 and Palmira for manuscript layout, PLM-SegFormer for palm-leaf damage segmentation; ihdia/sanskrit-ocr, pe-ocr-sanskrit (byte-level + SLP1 phonetic encoding for post-OCR correction), ByT5-Sanskrit (segmentation + post-correction SOTA, good starting checkpoint), RoundTripOCR (synthetic post-OCR pair generation — render gold, degrade with material profiles, run your own recogniser, pair with gold), OpenOCRCorrect (human-in-loop Indic correction UI), kraken/eScriptorium for historical hands. Linguistic resources: Digital Corpus of Sanskrit, GRETIL, SARIT, Cologne Digital Sanskrit Dictionaries, Sanskrit Heritage Platform, ambuda-org/vidyut, AI4Bharat IndicCorp, a chandas library for metre. Paleography: Bühler *Indische Palaeographie*, Ojha *Bhāratīya Prācīna Lipimālā*, Dani *Indian Palaeography*, Salomon *Indian Epigraphy*, Wujastyk on manuscripts; Unicode's Nandinagari proposal as a model for documenting style entries. Always verify current links, licences and checkpoints — these move.

## Security and rights

Any upload/inference endpoint needs authentication and rate limiting (abuse and GPU cost); a feedback endpoint without auth invites training-corpus poisoning. Manuscript images usually permit research use but not redistribution — ship manifests and download scripts, plus a per-item rights record, never the images.

## Deliverable style

When asked to produce the repo or documentation: give a concrete file tree, YAML/JSON schemas with real examples, an API contract, a phased roadmap, stratified evaluation plan, and an explicit limitations section stating which numbers are priors needing empirical validation by paleographers.

