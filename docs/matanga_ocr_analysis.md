# OCR Analysis Report: Mataṅga Bhāratam Manuscript

This report provides a comparative analysis of the first two pages of the handwritten manuscript **Mataṅga Bhāratam** (`Matanga_Bharatam.pdf`), contrasting the raw machine OCR outputs (Tesseract baseline fallback) with a human scholarly transcription (ground truth).

---

## 1. Document Medium & Quality Assessment

The system's newly integrated HTR document analyser processed the scanned image and returned the following quality profile:

- **Writing Medium Profile**: Classified as `degraded_paper` (aged manuscript paper, dark ink).
- **Luminance & Contrast**: Mean luminance $178.5$, contrast standard deviation $71.1$ (showing high background variation).
- **Degradation Score**: **`79%`** (reflecting heavy yellowing, bleed-through, ink degradation, and marginal smudges).
- **Physical Layout**: The presence of dark side borders and skew caused the vertical segmenter to fall back to a single page-wide text block, showing the limitations of simple horizontal projection models on historical scans.

---

## 2. Page 1 Analysis (Manuscript Title Page)

Page 1 serves as the cover/title page of the manuscript, containing a mixture of red-ink Devanagari titles and black-ink cursive English library notations.

### Visual Comparison (Page 1)

![Page 1 Scan Crop](/C:/Users/RSNK%20MAIN/.gemini/antigravity-ide/brain/0505eae5-c873-4b3c-8d6c-af84d590aabf/matanga_page1_screenshot.png)

### Transcription Mismatch Table

| Line/Element | Human Scholar (Ground Truth) | Raw Machine OCR Output | Error Taxonomy & Analysis |
| :--- | :--- | :--- | :--- |
| **Top Margin** | `२` (with underline & dot) | `ष = €~` | **E14 / E17**: The handwritten Devanagari numeral `२` and top edge smudges were misrecognized as a letter `ष`, operators, and symbols. |
| **Line 1 (auspicious)** | `॥ श्रीः ॥` | `\| सनक` | **E13 / E01**: The double danda `॥` was split into a single pipe. The conjunct `श्रीः` was completely misrecognized as `सनक`. |
| **Line 2 (numeral)** | `३` or `२` (centered) | `प वपत` | **E14 / E12**: The numeral was merged into surrounding noise, creating dummy characters `प वपत`. |
| **Line 3 (main title)** | `मतंग भरतम् नाम` | `ता सररतम लाम` | **E01 / E02**: Severe consonant confusion: `म` $\rightarrow$ `त`, `भ` $\rightarrow$ `स`, and `न` $\rightarrow$ `ल`. Underlining further distorted baseline projection. |
| **Line 4 (sub-title)** | `अभिनयलक्षणम् ॥` | `-उनिजनत्ट कणम्‌` | **E09 / E10**: Conjunct `क्ष` and local vowels were completely mangled. |
| **Lines 5–6 (notation)** | `Mathanga Bharatham.` <br> `B. 11546 (Comp. with 11526` | `/{-4 (०.1 6.24 4 2५ € * (5-46-९4. ^~ 45८` | **E15 / E17**: Cursive English metadata was forced through the Devanagari model, yielding gibberish. |

---

## 3. Page 2 Analysis (Main Text Folio 1)

Page 2 is the first page of the main treatise, containing a traditional Sanskrit invocatory verse (*Maṅgalācaraṇa*).

### Visual Comparison (Page 2)

![Page 2 Scan Crop](/C:/Users/RSNK%20MAIN/.gemini/antigravity-ide/brain/0505eae5-c873-4b3c-8d6c-af84d590aabf/matanga_page2_screenshot.png)

### First Three Lines Comparison

| Line | Human Scholar (Ground Truth) | Raw Machine OCR Output | Error Taxonomy & Analysis |
| :--- | :--- | :--- | :--- |
| **Margin / L1** | `२` (margin folio index) <br> `श्री वेंकटेशाय नमः ॥` | `[1 वत्र केर्यिशायलमद` | **E14 / E01**: The margin index `२` was read as `[1`. `श्री` became `वत्र`, and `वेंकटेशाय नमः ॥` was mangled into `केर्यिशायलमद` due to severe vowel and conjunct confusion. |
| **Line 2** | `अस्मद्वक्त्रपदपङ्कजयुग्मं भुक्तिमुक्ति` | `उस्न वन्कावद्र्प्मयु गमौ वाति <` | **E09 / E10**: Hand-drawn ligatures like `स्म`, `द्व`, `क्त्र`, and `ङ्क` were unrecognizable to the printed-font model. |
| **Line 3** | `सुकरं सुरसेव्यम् ॥` | `स्प्सेन्यम “` | **E04 / E08**: Vowels signs were dropped, `व्य` was confused with `न्य`, and the double danda `॥` was read as double quotes `“`. |

---

## 4. Key Differences: Machine Vision vs. Human Eyes

### 1. "Seeing font shapes" vs. "Reading script traditions"
The machine OCR treats letters as rigid templates, looking for visual template matches. It gets confused by **handwriting style variants**, stroke-width variations, and paper texture.
Conversely, the human eye reads the **gestalt flow of the hand**, recognizing characters by their stroke sequence, context, and script tradition (e.g. recognizing a danda `॥` as punctuation, rather than a quote `“` or brackets).

### 2. Conjunct Consonants Mismatches
Devanagari manuscripts are dense with conjuncts (ligatures). While a human immediately decodes the stacking of consonants (such as `स्म` or `ङ्क`), the regular OCR segmenter splits them or matches them to simple visual lookup alternatives (e.g., reading `पङ्क` as `प्म`).

### 3. Contextual and Lexical Awareness
When a human sees the sequence `अस्मद्वक्त्र...`, their knowledge of Sanskrit morphology immediately filters out visual noise. The regular OCR lacks this contextual ranking, printing nonsense words like `उस्न वन्काव` without flagging them as lexical errors.

### 4. Medium-aware Preprocessing
A human reader ignores paper yellowing, shadows, and ink bleed. The regular OCR segmenter treats shadows as lines of text, merging distinct columns or treating margins as actual letters.
