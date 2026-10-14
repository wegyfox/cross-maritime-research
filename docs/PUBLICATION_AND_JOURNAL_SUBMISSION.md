# Publication and journal-submission checklist

This public guide records the observed CROSS R2/PSP 1.0.0 release and its **14 October 2026** article submission to *Ocean Engineering*. UI details, prices and policies may change. This file contains no account-recovery information.

## Confirmed public outputs

- R2 scientific Dataset v1.0.0: https://doi.org/10.5281/zenodo.23345696
- PSP Unified LEAN Software v1.0.0: https://doi.org/10.5281/zenodo.23363285
- Public source: https://github.com/wegyfox/cross-maritime-research
- Manuscript: *From Pair-Context Effects to Network Bottlenecks in Mixed Autonomous and Human-Driven Ship Traffic: A Reference-Envelope Simulation Study*. Submitted on **14 October 2026** as a **Full length article**. Submission is not peer review, acceptance or journal publication.

## Publication gates: do not skip

1. Freeze the scientific core, experiments, configurations, seeds, numerical evidence and clear qualification scope. The CROSS results are synthetic and **not AIS-calibrated**, operational safety-certified or real-waterway capacity estimates.
2. Build ZIP releases with verified manifests, sizes, CRC, SHA-256 and provenance; Windows clean-install and bounded CLI checks; separate full GUI and security gates where relevant.
3. Review **exact-file** permissions: original author code Apache-2.0, specifically cleared synthetic data/docs CC BY 4.0, vendor MIT/BSD or other terms retained. Include component-rights map, upstream notices, security/checkpoint warning. Do not treat pickle/cloudpickle files from third parties as safe due to an adjacent SHA.
4. Publish sources, docs, tests, scoped CI and reproducibility pointers on GitHub. Do not mix large frozen research archives into Git history. Review secrets in tracked files **and history** before Public.
5. Deposit Dataset and Software as **separate Zenodo drafts**, verify upload filename, size/MD5, local SHA, author/ORCID, scoped license and description. Only owner-approved exact bytes may be published. Cite the **version DOI**, not just the all-versions concept DOI.
6. Update manuscript, cover letter, Highlights and Data/Code Availability with the actual public DOIs and GitHub status. Render Word to PDF; verify tables, figures, bibliography, declarations and truthful limits before submitting.
7. Obtain separate owner approval for GitHub public visibility, each Zenodo Publish, and final journal Submit. They are different irreversible actions.

## Zenodo pitfalls (observed)

- In the observed author form, ORCID bare number succeeded where a full URL produced `No valid scheme recognized for identifier`; approximate search results may show unrelated authors. Never select an unrelated person.
- Independent researchers can add their own independent-researcher label rather than affiliate with a similarly named third-party ROR organization.
- For relevant third-party assets use BSD **2-Clause "Simplified"**, not "2-Clause with views sentence". Multiple license tags do not put all files under all licenses.
- `Save draft` is allowed before upload succeeds. Preview/reserved DOI is **not** a registered public release. Published version files cannot simply be replaced.
- Name outer uploaded ZIPs as public releases, not internal `ZENODO_UPLOAD_REVIEW` work products. A filename-only rename preserves file bytes.

## Elsevier Ocean Engineering: observed 9-step new submission service

1. **Choose article type:** Full length article.
2. **Upload required files:** Word Manuscript, Cover letter, Highlights; competing-interest confirmation. The observed portal rejected filenames **over 64 characters**; a short name such as `CROSS_R2_Manuscript.docx` worked.
3. **Title/abstract/keywords:** verify auto-extracted text; the submitted abstract was shown as **181/200 words**.
4. **Author details:** explicitly mark the sole corresponding author and real CRediT roles. For independent work, this portal accepted a manually entered `Not affiliated to an organization` when the standard institution chooser rejected a free-text status. Never invent an employer/university.
5. **Funding:** no funder, if none.
6. **Publishing model:** separate Elsevier dialog offered **Subscription: Nothing to pay** or OA **€4,310 excluding taxes**, payable on acceptance. CROSS chose Subscription. Recheck actual price and rules for future submissions.
7. **Manuscript classifications:** select 3–5 relevant topics. CROSS selected marine autonomy, automatic control, stochastic processes, and other experimental approaches; no misleading full-scale/field measurements.
8. **Additional details:** CROSS chose NO to optional SSRN preprint, YES to research-data sharing, with Zenodo as repository, Original data, and R2 dataset **version DOI**. Program DOI was documented separately in the manuscript.
9. **Review and submit:** all previous tasks Done; author personally reviews ethics, copyright, originality, funding and authorship statements. `Complete your submission` **actually sends** the paper. Preserve portal confirmation.

**Subscription does not prohibit freely hosted independent software and research data.** It may restrict sharing of the article's accepted manuscript and publisher's final PDF. The 2026 form stated **24 months** for certain public repository posting of an accepted manuscript. Verify the current publishing agreement separately for preprint, accepted manuscript and version of record.

## After submitting

The **new portal** displayed `Your manuscript has now been submitted`, and its dashboard `Submitted`. The `Check editor progress` link opened the **separate Editorial Manager / Aries** login. A registration attempt there returned `Duplicate Registration` because an account already existed: do not create a second account or submit the manuscript again. Instead sign in to the existing account or use recovery when needed.

Editorial Manager personal expertise required **5+ personal classifications**, distinct from the new portal's **3–5 manuscript classifications**. The `Submit` control inside this classification picker merely saves profile selections.

Wait for an actual manuscript tracking ID and editorial correspondence; do not invent one. See [release status](RELEASE_STATUS.md), [license scope](../LICENSE_SCOPE.md), and [security notes](../SECURITY.md).

The separate owner-only Russian full playbook, account-related decisions and screen-specific corrections are stored privately, not on public GitHub.
