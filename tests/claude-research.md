How Teams Sanitise Documents Before Sharing With AI Vendors
20 Nov 2026 · @Jerin
Summary
Published practice has converged on one pattern: detect inside the perimeter with a hybrid of rules, NER and LLMs; replace entities with consistent typed pseudonyms; keep the mapping in-house; and put a human or a sample check behind it, because no detector is complete. Nobody credible claims automated redaction is safe on its own. Presidio's own docs say there is no guarantee it finds everything, and the French Cour de cassation still has every pseudonymised judgment read by a person.
Five findings matter most for AutoDoc:
1.  Hybrid detection beats any single method. Regex with checksums handles structured identifiers, and models handle names in context. Recall, not precision, is the metric that counts.
2.  Distinct, typed pseudonyms preserve LLM utility. Generic redaction destroys it. A September 2026 study found that replacing every entity with the same tag, or deleting it, caused the largest performance drops. Stable labels like LOC-1 came close to the unmodified baseline.
3.  Removing names does not stop re-identification. Research from ETH Zurich and the RAT-Bench team shows LLM attackers infer identity from indirect clues, and that even the best tools leave real risk. The emerging defence is an adversarial LLM check: ask a model who the subject is, then fix what gives it away.
4.  Most real-world leaks are file-format failures, not detection failures. Text under black boxes, OCR layers and metadata keep exposing data, as in the 2025 Epstein files release. Rendering fresh output is the reliable fix.
5.  For credit agreements specifically, public filings are the main re-identification risk. Public borrowers file their credit agreements on EDGAR. Each agreement's confidentiality clause also governs whether disclosure to a vendor is allowed at all.
The AutoDoc design from yesterday matches this practice closely. The research suggests seven additions (see the implications section). The most important: keep labels distinct per entity and settle the surrogate-vs-token question with the vendor early, since both are used successfully; make the adversarial re-identification pass a first-class stage; add a proper recall-sampling protocol borrowed from eDiscovery; and evaluate an off-the-shelf detector before building your own.
Tooling landscape
The market splits into open-source building blocks you assemble yourself and commercial products that bundle detection, replacement and a UI. None of them understands credit agreements out of the box, so each would sit inside AutoDoc's detection stage as one detector rather than replace the pipeline.
Tool
Type
How it works
Relevance to AutoDoc
Microsoft Presidio
Open source (Python)
Recognisers combine regex, NER (spaCy by default), checksums and context words. Operators: replace, redact, mask, hash, encrypt. Ships an evaluation framework (presidio-research).
The de facto default and a good detector-framework reference. Its own docs say detection is never guaranteed complete and recommend weighting recall over precision. Independent tests on messy text report recall well below vendor claims (Grepture).
OpenAI Privacy Filter
Open-weight model (Apache 2.0, April 2026)
1.5B-parameter token classifier, about 50M active, run locally. Eight categories including person, address, email and account number. Reported F1 96% on PII-Masking-300k. Handles long inputs in one pass.
A strong, cheap local detector for personal data. It does not cover company names or deal-specific identifiers, so it fills only part of the job.
Tonic Textual
Commercial
Multilingual NER across PDFs, Office files and images. Per-entity choice of redact or synthesise, with replacements consistent across a dataset. Has an LLM "grouping synthesis" mode that keeps related entities coherent, and a financial documents guide.
Closest commercial match to the "consistent surrogate" requirement. Its consistency is per value, not per resolved entity, so corporate families and aliases would still need AutoDoc's registry.
Private AI
Commercial, on-prem container
Transformer-based contextual detection, 50+ entity types, numbered tokens (NAME_1) or synthetic replacement, reversible tokenisation. Also scans embedded images.
Deploys inside the firm, which matters for a bank. Same gap on deal-specific entities.
Google Sensitive Data Protection
Cloud service
Deterministic encryption (AES-SIV), format-preserving encryption, or hashing, keyed via KMS. Same input and key give the same token, so references stay consistent.
A useful model for how to mint stable pseudonyms and keep them reversible under a key. Sending raw text to a cloud API may itself be off-limits.
AWS Textract + Comprehend + A2I
Cloud services
Textract OCR, Comprehend PII detection, rule-based redaction in Lambda, and A2I for human review of low-confidence results.
AutoDoc already uses Textract. AWS's reference architecture for redaction is essentially detect, then send flagged items to a human, which matches the review design.
Legal-specific tools (e.g. LEGALFLY, Definely Draft)
Commercial
Anonymise party names, addresses, amounts and dates in contracts before AI processing, keeping legal citations.
Show the market expects contract-aware anonymisation, but these are built for lawyers using AI tools, not for producing vendor training data at scale.
Build vs buy. The detection layer is increasingly a commodity, and Privacy Filter plus Presidio cover personal data well. What no product provides is the part that makes this hard for credit agreements: resolving corporate families and roles into one consistent registry, a keep-list of role-based defined terms, and an approval and egress process. That is where AutoDoc's own investment should go.
Architecture patterns teams use
Four patterns recur across vendor guides, engineering blogs and papers.
1. Hybrid detection: rules first, then models. Regex with checksums handles structured identifiers with near-perfect precision. Statistical or transformer NER handles names. An LLM handles context and ambiguity. Several sources describe this division of labour as the best overall performer, at the cost of more maintenance (legal SaaS security course, Ertas benchmark). A 2025 paper combining regex with context-aware LLM prompting reported much higher recall than either NER or a zero-shot LLM alone (RECAP, arXiv 2510.07551).
2. Reversible pseudonymisation with the map kept inside. The standard flow is: detect, replace with tokens, send only the clean text, and restore real values in the response using a map that never leaves (DZone, Presidio as a guardrail). Recommended details include deterministic (keyed or hash-based) mapping, longest-match-first replacement for nested entities, and an exclusion list of legal terms that look like names. The redaction map is treated as a crown-jewel secret, because anyone with it can reverse every document.
3. Consistency scoped deliberately. Tools make replacements consistent within a chosen scope: Tonic within a dataset, Google within a key and context. The CON-QA contract paper chose the opposite trade-off: one surrogate per entity within a session, with the mapping discarded afterwards so the same company gets different names across sessions and cannot be linked. That is the same "per document set by default" choice proposed for AutoDoc.
4. A gateway that everything outbound must pass through. Most enterprise write-ups put redaction in a central proxy or gateway rather than trusting each application to do it, and log what was redacted and where it went (Arthur, Wavect). Several also warn that logging and observability stacks capture raw text before redaction runs, and need their own filter.
Typed placeholders vs realistic surrogates. Both are in use. Tokens like PERSON_001 are easy for reviewers to spot and are what most tools default to. Realistic surrogates (one fake company name per real one) read naturally and were used in CON-QA, which reported about 99% entity coverage and 98% correct restoration. What the evidence rules out is collapsing everything into one generic tag (see the next section).
What the research says
The academic picture is consistent: detection recall is imperfect, removing names is not anonymisation, and the most effective new defence uses an LLM as the attacker.
Finding
Evidence
What it means for AutoDoc
LLMs infer identity from indirect clues, and they beat industry anonymisers when used adversarially
Staab et al., ETH Zurich, ICLR 2025: an anonymiser that repeatedly asks an LLM to infer personal attributes, then rewrites what gave them away, outperformed Presidio and Azure Language on both privacy and utility across 13 models (paper)
Strong support for the adversarial re-identification stage in the design
Even the best tools leave real re-identification risk, especially from indirect identifiers and non-standard formats
RAT-Bench (Krčo, Meeus, de Montjoye et al.), February 2026, which scores tools by attacker success rather than entity recall (paper)
Measure re-identification, not only entity recall, in the evaluation harness
Generic tags and deletion hurt LLM performance most; distinct per-entity labels keep it close to baseline
Deußer et al., September 2026: on a multi-step reasoning benchmark, unique labels scored 0.70 against 0.73 unmodified; generalisation scored 0.58 and redaction 0.52 (as reported by HAQQ; legal documents were not tested)
Confirms one distinct label per entity. Supports leaving role terms like "the Borrower" intact.
Choosing a replacement by testing whether an LLM can guess the original keeps utility with under one point of extra risk
arXiv 2412.12928, on the Text Anonymization Benchmark of court cases (paper)
A cheap way to pick safe surrogates if realistic names are used
Contract-specific pseudonymisation works with local detection and per-session surrogates
CON-QA (arXiv 2509.19925): GLiNER detection, local surrogate generation, a cloud LLM on anonymised text. About 99% coverage vs 76% for a baseline; 98% restoration accuracy; small sample of 50 queries (paper)
Closest published analogue to this use case. Also re-runs detection on the model's output.
A hybrid rules + ML system on financial documents reached 94.7% precision and 89.4% recall
Mishra et al., Scientific Reports, July 2025, mostly on synthetic data (paper)
Even a good hybrid misses about 1 in 10 entities. Verification and review are not optional.
Ambiguous names (that are also common words) cut recall by 20–40% across LLMs, NER and commercial tools
"Can LLMs Really Recognize Your Name?" (review)
Company stems that are ordinary words ("Summit", "Apex") need registry-based matching, not detector judgement
General NER over-redacts; domain-tuned models do much better
Comparison across 231 configurations (arXiv 2507.11862)
Budget for fine-tuning or few-shot prompting on credit-agreement text, using the golden set
The common lesson: a detector's recall on a benchmark says little about re-identification risk on your documents. Both have to be measured on your own data.
Financial services and legal practice
Credit agreements of public borrowers are already public. Public companies routinely file their credit agreements as exhibits to 8-K and 10-K filings on EDGAR, with the full parties clause, agent and lender names (example: a 2023 Texas Capital Bank–agented agreement). Filers may omit only information that is not material and that is either harmful to disclose or treated as confidential, marked with [***] (example notice). Financial covenants are generally material, so they usually remain. This means a pseudonymised agreement for a public borrower can often be matched back to its public filing from the covenant text alone. That residual risk should be accepted knowingly, not discovered later.
The agreement's own confidentiality clause decides what can be shared. The standard "Treatment of Certain Information; Confidentiality" section binds the agent and lenders, with listed exceptions: affiliates and their advisors, assignees and participants, rating agencies, regulators, the CUSIP bureau, and sometimes "service providers to the Administrative Agent and the Lenders in connection with the administration" of the agreement (examples on Law Insider). Whether an AI vendor building models fits any of these exceptions varies by agreement. Mayer Brown notes that private-credit agreements in particular restrict disclosure tightly and may bar disclosure to named competitors, with real liability for breach (Mayer Brown, 2025). The legal analysis is likely to be: information that no longer identifies the borrower may fall outside the clause, which raises the bar on how well identity is removed.
Banks lean on synthetic data for sharing. J.P. Morgan AI Research publishes synthetic datasets for research and model development, and has proposed a six-level framework for grading the privacy of financial synthetic data (J.P. Morgan, framework review). This is mostly tabular and transaction data, not long legal documents, but it suggests a complementary option: synthetic credit agreements built from templates and public filings for the bulk of training, with a smaller sanitised real set for evaluation.
Law firms use pseudonymisation plus contracts plus policy. Firm guidance treats name-swapping with a retained key as pseudonymisation, which data protection law still treats as personal data, so the receiving tool must still be approved (example). Firms also lean on DPAs and private deployments (for example, Azure OpenAI under the Azure DPA) as the primary control, with redaction as an extra layer (Softomate).
Courts pseudonymise at scale, with humans in the loop. France's Cour de cassation built ML-based pseudonymisation to publish court decisions as open data. Its Judilibre database held about 910,000 pseudonymised decisions (La Gazette). Because rules and models alone were not reliable enough, every published decision was required to go through human review (Acteurs publics, 2019). It is the largest public precedent for long legal documents, and it chose mandatory review.
Redaction failures are usually file-format failures. High-profile leaks keep coming from "redactions" that only drew boxes over text: hidden text, OCR layers and metadata left in place, as in the December 2025 Epstein files release and Meta's FTC trial exhibits (Redactable on Epstein, on Meta). Scanned PDFs are a particular trap, because OCR adds an invisible text layer beneath the image (PDF Tools).
Evaluation, verification and human review
Measure recall first, on your own documents. Presidio's evaluation guide recommends an F-beta score with beta = 2, which weights recall above precision, because a missed entity is a leak and an extra redaction is only noise (Presidio evaluation). The presidio-research package provides synthetic data generation and per-recogniser precision/recall analysis (PyPI). Every serious source warns that vendor benchmark numbers do not transfer: one independent test of general tools on messy real text found aggregate recall of 57–73% (Grepture).
Add a re-identification metric. RAT-Bench and the Staab et al. paper both evaluate by asking an LLM attacker to recover identity from the output. For credit agreements, the practical version is: give a strong internal model the sanitised text and ask it to name the borrower, the agent bank and the sponsor. Track the success rate per document and per release.
Check the output, not just the detector. Several guides recommend running detection again on what leaves, not only on what comes in (CON-QA, Openlayer). For files, verification means checking the delivered artifact itself: copy-paste and text extraction, the OCR layer, metadata and annotations (Nutrient, Syncfusion).
Route by confidence, sample the rest. AWS's reference architectures send low-confidence predictions to human reviewers through A2I, and also send a random sample of confident ones to measure quality (AWS ML blog). eDiscovery, which has decades of practice reviewing large legal document sets, uses the same two tools: second-level review of first-pass calls, and statistical sampling with documented methodology (Logikcull). Epiq describes estimating elusion — how much was missed in the set presumed clean — from a random control set of around 500 documents (Epiq).
Translate elusion to this problem. For AutoDoc, the equivalent is: randomly sample pages (or chunks) from approved sanitised output, have a reviewer who did not do the first review inspect them for anything identifying, and report the miss rate with a confidence interval. That gives Compliance a number to sign off against, rather than an assurance.
Implications for the AutoDoc design
What the research confirms in the existing design
Design choice
Supporting practice
Hybrid detection: checksummed patterns, NER, chunked LLM
Recommended by most sources; best results in published comparisons
One distinct label per entity; leave role-based defined terms alone
Distinct labels preserve LLM utility; generic tags and deletion damage it most
Registry scoped to the document set; fresh labels across deals
Same trade-off as CON-QA's per-session surrogates, chosen to prevent linkage
Mapping vault that never leaves
Universal: the map is treated as the key to every document
Render fresh output; never redact in place
Real-world leaks are overwhelmingly hidden text, OCR layers and metadata
Adversarial re-identification pass
Staab et al. and RAT-Bench show LLM attackers are the right test
Mandatory human review driven by exceptions
Cour de cassation, AWS A2I and eDiscovery all keep a human gate
Pseudonymisation, not anonymisation; Legal signs off on residual risk
Law-firm and data-protection guidance draws the same line; EDGAR makes it concrete
What to change or add
1.  Make re-identification a tracked metric, not just a gate. Report per document and per release the share of documents where an internal LLM attacker correctly names the borrower, agent or sponsor. Include an "EDGAR match" test: can the sanitised covenant text be matched to a public filing? Use these numbers in the Legal sign-off.
2.  Add an elusion sample to the review process. After approval, a second reviewer inspects a random sample of pages. Report the miss rate with a confidence interval. This is borrowed directly from eDiscovery practice and gives Compliance a defensible number.
3.  Run detection on the vendor's outputs too. If the vendor returns extracted covenants, models or reports, scan them for anything that maps back to the registry before they re-enter AutoDoc.
4.  Evaluate off-the-shelf detectors before building. Benchmark OpenAI Privacy Filter (local, Apache 2.0) and Presidio with custom recognisers against the golden set. Consider a commercial on-prem option (Private AI, Tonic Textual) only if the open models miss too much. Keep the registry, keep-list and policy engine in-house either way; no product covers them.
5.  Decide surrogate vs token with evidence. Both work. If the vendor wants realistic names, generate surrogates locally and choose each one by testing that an LLM cannot guess the original from it (the arXiv 2412.12928 method). Avoid any single generic tag.
6.  Consider a synthetic tier. For bulk training data, synthetic credit agreements generated from templates and public filings carry no client risk. Reserve sanitised real agreements for evaluation and fine-tuning where realism matters. This mirrors how banks already use synthetic data for vendor collaboration.
7.  Get the confidentiality-clause analysis done per agreement. Practice in private credit shows these clauses vary and can be strict. Build a "cleared for sharing" flag into intake so an agreement cannot enter the pipeline until Legal has cleared it.
Sources
Research papers
• Staab et al., Language Models are Advanced Anonymizers, ICLR 2025
• Krčo et al., RAT-Bench: A Comprehensive Benchmark for Text Anonymization, 2026
• CON-QA: privacy-preserving contract question answering, arXiv 2509.19925
• LLM-based replacement with inference-attack selection, arXiv 2412.12928
• Mishra et al., Hybrid rule-based NLP and ML for PII in financial documents, Scientific Reports 2025
• RECAP: hybrid regex + LLM PII detection, arXiv 2510.07551
• Domain-tuned vs general NER for PII, arXiv 2507.11862
• Can LLMs Really Recognize Your Name? (review)
• Deußer et al., On the Impact of Anonymization on LLM Performance, 2026, as summarised by HAQQ
Tools and vendor documentation
• Microsoft Presidio evaluation and presidio-evaluator
• OpenAI Privacy Filter
• Tonic Textual and financial documents guide
• Private AI
• Google Sensitive Data Protection pseudonymisation
• AWS human-in-the-loop redaction and end-to-end IDP with A2I
Practice and guidance
• Redaction pipelines for cloud AI (legal SaaS security)
• Grepture: Presidio is not enough
• Wavect: PII redaction before LLM prompts
• Mayer Brown: confidentiality restrictions in loan documents
• Credit agreement confidentiality clauses (Law Insider)
• Example credit agreement on EDGAR
• J.P. Morgan AI Research synthetic data
• Cour de cassation open data (La Gazette) and Acteurs publics
• Epiq: sampling in AI-assisted discovery and Logikcull: QA for privilege review
• Redactable: Epstein files failure and PDF Tools: hidden layers
Vendor benchmark figures (Tonic, Privacy Filter, Private AI) are self-reported.
