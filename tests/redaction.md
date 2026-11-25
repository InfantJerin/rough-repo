# Client/Server PDF Redaction & Pseudonymization — Summary

**Date:** 2026-11-25
**Context:** Replace sensitive names in credit agreements (e.g., "Acme Holdings LLC") with pseudonymous labels ("Borrower 1", "Guarantor 1") **inside the PDF content itself**, preserving the original layout. Stack: React frontend, Python backend.

---

## 1. Key conclusions

### PDF.js is not a redaction tool
- PDF.js is a renderer/viewer. Its highlight/annotation editor adds objects **on top** of the page; it cannot remove or rewrite page content.
- Drawing a black box (PDF.js annotation or pdf-lib rectangle) leaves the original text underneath — still selectable and extractable. This is the classic redaction failure.
- **Right role for PDF.js:** preview, selection and reviewing the redaction boxes in React. The actual content rewrite belongs in the backend.

### "Replace text, keep layout" breaks into three problems
A PDF has no text flow — only glyphs drawn at fixed positions.

1. **Locate** — get character-level bounding boxes for the target strings.
2. **Remove only those glyphs** — a single text-drawing command (`Tj`/`TJ`) often holds the whole line, so it must be split, keeping neighbouring text at its exact original positions.
3. **Draw the replacement label in the same box**
   - **Missing glyphs:** embedded fonts are usually subsets; characters like "B" or "1" may not exist. Need a fallback font (standard Helvetica or an embedded font).
   - **Width mismatch:** nothing reflows. Leave a gap after the label or scale it to fit — this is *why* the layout stays intact.

PyMuPDF's approach is exactly steps 2 + 3: apply the redaction (character-level removal), then insert text into the same rectangle.

### Licensing reality check
- **PyMuPDF / MuPDF** is AGPL — but Artifex **dual-licenses** it commercially. AGPL alone doesn't rule it out if GS buys (or already holds) a commercial licence.
- **Porting PyMuPDF/MuPDF logic** into another library by reading its source risks being treated as a derivative work. Building from the PDF spec and the target library's own APIs is clean; translating MuPDF code is not. Needs a clean-room approach and sign-off from the open-source office. *(Not legal advice.)*

---

## 2. Library options

| Option | Licence | How it fits | Effort / risk |
|---|---|---|---|
| **PyMuPDF** (commercial licence) | Commercial (Artifex) | Native redact + insert-text workflow; closest to a ready-made solution | Low effort; procurement cost |
| **pypdfium2** (PDFium) | Apache-2.0 / BSD-3 | Char boxes via `PdfTextPage`; remove/re-create text objects via raw `FPDFPageObj_*` / `FPDFTextObj_*`; rewrite with `FPDFPage_GenerateContent` | Medium–high: PDFium works at whole-text-object granularity, so splitting and re-placing kept text is fiddly |
| **pikepdf** (qpdf) + **pdfplumber** | MPL-2.0 / MIT | Edit content-stream operators directly; pdfplumber for locating text | High: you handle font encodings, CMaps and ToUnicode yourself; maximum control |
| **Aspose.PDF for Python** | Commercial | Text-search tools that find and replace text in place | Low effort; licence cost |
| **Apryse (PDFNet / WebViewer)** | Commercial | Redaction with custom replacement text over the cleared area; also client-side WASM | Low effort; licence cost |
| **MuPDF.js / open-redact-pdf** (browser WASM) | AGPL / young OSS project | True client-side redaction | AGPL issue for MuPDF.js; open-redact-pdf is immature |
| **Rasterize fallback** | Any | Render page → paint boxes → rebuild PDF from images | Secure but loses selectable text unless OCR'd; doesn't meet the "replace with label" goal cleanly |

---

## 3. pypdfium2 implementation sketch

1. Open the document; for each page build a `PdfTextPage` and find char boxes for every target string.
2. Iterate page text objects (raw PDFium API).
3. For each text object overlapping a target:
   - capture its font, size, transform and per-character positions;
   - remove the object (`FPDFPage_RemoveObject`);
   - re-create the **kept** segments as new text objects at their original positions (same font where possible).
4. Add the pseudonym label as a new text object in the target box (fallback font if glyphs are missing; left-align, scale if too wide).
5. `FPDFPage_GenerateContent` → save a **fresh full copy** (never incremental).

---

## 4. Leak checklist (applies to any library)

- [ ] **No incremental saves** — they append changes and keep the original bytes in the file. Do a full rewrite with garbage collection.
- [ ] Strip **metadata** (Info dict + XMP).
- [ ] Check **bookmarks/outlines, form fields, annotations, attachments**.
- [ ] Handle **headers/footers and signature blocks**.
- [ ] Handle **scanned pages** — hidden OCR text layer and image pixels.
- [ ] **Verify output:** run text extraction on the redacted PDF and assert none of the original names remain.

---

## 5. Action items

| # | Action | Owner | Priority |
|---|---|---|---|
| 1 | Ask procurement / open-source office whether GS already holds a commercial licence for **PyMuPDF/MuPDF, Apryse or Aspose** | Jerin | High — this week |
| 2 | Confirm with the open-source office the rules for a **clean-room implementation** (no reading MuPDF source) | Jerin | High |
| 3 | **Prototype pypdfium2** on 2–3 real credit agreements, starting with the ugliest one (1–2 days) | Jerin / team | High — in parallel with #1 |
| 4 | Evaluate the **subset-font / missing-glyph** problem and choose a fallback-font strategy | Prototype owner | Medium |
| 5 | Decide the **width-mismatch policy** (leave gap vs scale-to-fit) | Jerin | Medium |
| 6 | Build an automated **verification step** (text extraction + assert no original names; metadata/attachment checks) | Team | High — before any production use |
| 7 | Define the architecture: **React + PDF.js** for preview/review of boxes → **Python backend** for the content rewrite | Jerin | Medium |
| 8 | If licensing comes through, compare **commercial-library output vs the pypdfium2 prototype** and pick one | Jerin | After #1 and #3 |

---

## 6. References
- MuPDF.js redaction how-to: https://mupdfjs.readthedocs.io/en/latest/how-to-guide/annotations/redactions/index.html
- MuPDF WASM docs: https://mupdf.readthedocs.io/en/latest/mupdf-wasm.html
- open-redact-pdf browser integration: https://fabifont.github.io/open-redact-pdf/guides/browser-integration
- Apryse WebViewer redaction: https://docs.apryse.com/web/guides/redaction/redact/
- Nutrient (PSPDFKit) programmatic redaction: https://www.nutrient.io/guides/web/redaction/programmatically/
