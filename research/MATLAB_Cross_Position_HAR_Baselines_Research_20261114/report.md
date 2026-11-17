# MATLAB-Ready Recent Baselines for Cross-Position HAR

**Research date:** 14 November 2026  
**Question:** Which recent transfer-learning/domain-adaptation methods are usable as baselines for the repository's MATLAB cross-position HAR workflow, with public MATLAB `.m` code?  
**Scope assumption:** The target experiment is the within-dataset, body-position-to-body-position task represented by `demo.m` and the repository's DSADS, OPPORTUNITY, and PAMAP2 MAT files.

## Executive Summary

**Recommendation: start with DAPCA (Domain Adaptation Principal Component Analysis, 2023) as the primary new baseline.** Its authors publish a MATLAB implementation, and its interface explicitly takes labeled source feature vectors and unlabeled target feature vectors, learns a linear shared representation, and returns projections for both domains. That is a close structural match to this repository: each body position is represented by 81 pre-extracted features, the domains share activity labels, and the workflow already supplies source labels and target sensor data. This makes DAPCA a defensible *candidate* baseline for the current task. It is not, however, a published cross-position HAR result; its performance on these MAT tasks must be measured before drawing conclusions. [S01][S02][S03][S04][S05]

**ICSC (Incremental Confidence Samples into Classification, 2022) is a secondary option.** Its paper describes unsupervised adaptation by progressively pseudo-labeling confident target samples, and a public MATLAB repository contains `ICSC.m`. Its demo is on image-domain data, not wearable HAR. The public function also receives target labels to print accuracy during iterations, even though its adaptation path uses pseudo-labels; for a clean target-label-blind experiment, move scoring outside the method and recode activity IDs to `1:C`. [S06][S07][S08]

The newer papers with the closest HAR wording are not good fits for the current constraints: the 2023 CNN+DANN/DDC+OS-ELM method is explicitly cross-position but uses a few labeled target examples and no public MATLAB release was verified; ActiLabel evaluates cross-location HAR but has no verified MATLAB `.m` release. The existing SOT implementation is Python, and its paper's principal evaluation is cross-dataset with a smaller feature setup, so its reported results should not be treated as directly comparable to the current within-dataset position tasks. [S09][S11][S16][S17]

## Introduction: What Counts as Usable Here?

The repository describes three position-resolved datasets, with 81 features per body part, and `demo.m` extracts one position's feature columns as source and another position's columns as target. Its example pairs include right arm to left arm, right leg to left leg, and arm to torso. The original STL paper defines this as cross-position activity recognition and evaluates within-dataset transfer on DSADS, OPPORTUNITY, and PAMAP2. Therefore, a suitable baseline should support two numerical feature matrices of the same width, source activity labels, unlabeled target features during adaptation, and a shared activity-label space. [S01][S02]

I treated “recent” as roughly 2020 onward, searched for methods with code, and required direct evidence of a MATLAB `.m` implementation before ranking one. I separately checked whether the method's learning setup matches the current protocol. A method being MATLAB-based or HAR-related by itself was insufficient. An application-fit statement below is an inference from the method's published input assumptions and the repository's data shape; it is not a claim that the method has already been validated on these tasks.

## Main Analysis

### Finding 1 — DAPCA is the strongest fit to the existing feature workflow

DAPCA learns a linear projection from labeled source data and unlabeled target data. Its paper frames the method as a domain-adaptation extension of supervised PCA, intended to reduce representation differences between the labeled source and unlabeled target while retaining source class information. The authors' repository contains `DAPCA.m`, explicitly says the MATLAB file can be copied into a project, and the function accepts `XX`, source labels, `YY`, and a requested component count; it returns a basis and projected source/target matrices. [S03][S04][S05]

That interface maps naturally onto the current experiment: select the 81-column feature block for the source position and target position, pass source labels but withhold target labels, use the projected source features to fit the chosen classifier, and evaluate predictions on the projected target features. Since both positions are extracted from the same dataset and share the same activity labels, this is a closed-set, same-feature-width adaptation setup. The repository's `.mat` files and the STL paper support those data assumptions. **This compatibility judgment is an inference**; neither the DAPCA paper nor its repository reports this exact body-position benchmark. [S01][S02][S03][S05]

Why first: it has the cleanest target-label boundary of the verified choices, it does not require a deep network or raw window sequences, and it can be inserted before the existing MATLAB classifier with modest pipeline changes. It is recent enough to be a meaningful newer feature-level baseline (published in *Entropy* in 2023), but it should be described as a generic linear UDA method applied to CPAR, not as a newly established HAR state of the art. [S03][S05]

Limitations: published evidence does not establish that DAPCA beats STL or other baselines on DSADS/OPPORTUNITY/PAMAP2 cross-position tasks. Its authors position it as a base linear method and evaluate it outside this exact benchmark. The GitHub repository declares GPL-3.0, so preserve attribution and check that license before redistributing a modified copy with your project. Treat component count and other DAPCA settings as experimental choices selected without target test labels. [S03][S04]

### Finding 2 — ICSC is a viable second MATLAB method, with two code-level safeguards

ICSC was published in 2022. Its method progressively admits confident pseudo-labeled target samples and adjusts the trade-off between transferability and class discrimination. The author-linked MATLAB repository includes `ICSC.m`, `ACDA.m`, a demo, and an MIT license. The method is a feature-level classifier adaptation approach, so in principle it can consume the same source/target feature vectors used in this project. [S06][S07][S08]

However, the repository demo and published reported results use image-domain benchmarks rather than wearable activity data, so its cross-position performance is unknown. There are also two implementation details to handle before a clean experiment. First, `ICSC.m` accepts `Yt` and uses it to compute and print target accuracy inside the iteration loop. The adaptation uses target pseudo-labels, but passing ground-truth `Yt` into the method function blurs the evaluation boundary. Remove/move the scoring line so target truth is used only after predictions are finalized. Second, the code loops over class IDs `1:C` when forming class means; if the activity IDs are not consecutive integers starting at 1, remap them consistently before calling it. [S07][S08]

I rank ICSC second rather than first because these safeguards require a small wrapper/code adjustment and the published validation domain is far from wearable sensor placement. If you want only methods with a direct cross-position evaluation in their original paper, do **not** count ICSC as such; it is a new method candidate to evaluate on your task. The repository's MIT license is permissive, but retain the method citation and code attribution. [S06][S07]

### Finding 3 — The most directly related recent HAR papers fail the `.m` or protocol gate

Xu et al. (2023) is the closest recent paper by task title: it studies cross-person and cross-position wearable HAR and combines a CNN feature extractor with DANN or DDC adaptation and an OS-ELM classifier. The paper reports using a few annotated target samples. That is a different learning protocol from the original no-target-label CPAR formulation, and the method is a deep sequence/representation pipeline rather than a drop-in feature-level transform for the precomputed 81-column matrices. I found no public MATLAB `.m` implementation in the paper/code searches. It is therefore useful related work, but not a qualifying baseline under the current constraints. [S09][S01][S02]

ActiLabel (2023) is another task-relevant paper: it reports cross-location activity learning by matching structural graphs and inferring target labels without manual target labels. Its experiments include PAMAP2 cross-location scenarios. I could not verify a public MATLAB `.m` implementation, and its graph construction, clustering, and assignment pipeline is not a ready-to-call feature transform in this project. It fails the strict code-availability gate, even though its problem setting is relevant. [S11]

Chang et al. (2020) provide a rigorous study of unsupervised adaptation for wearable-position diversity over multiple datasets and position pairs. This is valuable context for the paper's motivation and protocol, but it does not by itself provide a verified MATLAB implementation of a recent drop-in baseline for the current engineered feature matrices. Its analysis also emphasizes that results depend on dataset structure, class match, and target data availability; this supports testing the methods on the actual position pairs rather than importing headline numbers. [S10]

### Finding 4 — Avoid recency-by-toolbox and cross-protocol comparisons

DomainATM is a MATLAB toolbox published in 2023, but its feature-level menu contains established methods such as CORAL, TCA, JDA, TJM, GFK, and OT. It is useful as a MATLAB source for implementations or as an integration aid, but the toolbox's publication date does not make those underlying methods new. Since this repository already has MATLAB implementations of CORAL, TCA, and GFK, adding DomainATM alone would not satisfy a request for a recent algorithmic baseline. [S14][S15]

SOT (2021) is relevant HAR literature and its idea is not dismissed here. But the current project implementation is Python (`sot/main.py` imports `SOT.py`), violating the MATLAB-only implementation requirement. Its paper's benchmark uses 38 features and four common activities for cross-dataset experiments, while the current MATLAB demo's highlighted tasks transfer among positions within a dataset using 81 features per body part. It is possible to adapt or port SOT, but it is not a ready MATLAB baseline for this study and its published scores do not validate the current experiment. [S16][S17][S01][S02]

The 2023 SWL-Adapt paper studies cross-user wearable HAR and has an official implementation, but its repository specifies Python and PyTorch. It is not a MATLAB `.m` option, and cross-user transfer is not the same as the position-to-position problem here. [S12][S13]

## Synthesis & Insights

The hard constraint is not simply “an algorithm that can be coded in MATLAB.” The useful intersection is: recent enough to strengthen the paper's baseline set, a published source-to-target adaptation method, target labels withheld during learning, compatible with same-width engineered feature vectors, and a public MATLAB implementation. DAPCA is the only option in this scan that cleanly meets all of those *implementation and input-shape* conditions. ICSC also has MATLAB code and a plausible source/target fit, but needs label-handling safeguards and has no HAR evidence. Neither should be described as an established CPAR benchmark result until run on the repository's actual tasks. [S01][S02][S03][S05][S06][S07][S08]

The strongest domain-specific 2023 methods solve different versions of the problem: Xu et al. permit a few labeled target examples and use a deep architecture, while ActiLabel solves structural label mapping and has no verified `.m` release. They may inspire a future method port or a separate few-shot experiment, but inserting them into the current baseline table would silently change the protocol or implementation requirement. [S09][S11]

The paper can make a careful contribution by evaluating DAPCA (and optionally ICSC) on the same fixed position pairs used by STL, alongside the existing MATLAB baselines. The novelty claim should be the evaluation on this CPAR protocol, not that DAPCA itself is new or that the published generic results transfer to HAR. Report per-task results and an aggregate across tasks; accuracy plus macro-F1 is safer than accuracy alone for class imbalance, consistent with HAR studies in this area reporting both accuracy and F1-family metrics. [S01][S11][S12] These are recommendations, not claims of measured performance.

## Limitations & Caveats

This is a code and protocol audit, not a benchmark run. I did not execute DAPCA or ICSC in MATLAB or claim they improve accuracy. Public code availability can change; “not verified” means I did not locate a public MATLAB `.m` implementation in the available paper and repository search, not proof that no private or unindexed code exists. DAPCA and ICSC were not validated by their cited papers on this exact body-position benchmark. Results must be generated on the local MAT data before selecting a winning baseline.

The repository's demo appears to use the full target feature matrix while evaluating those same target rows. That is a transductive UDA protocol: target features may be exposed to adaptation, but target labels must remain unavailable until final scoring. State this explicitly. If your intended paper protocol is inductive, reserve separate target adaptation and target test sets, and do not fit normalization or the adaptation transform using target test features. Keep the same split, preprocessing, classifier convention, and random seeds across methods. [S02]

## Recommendations

1. **Implement/evaluate DAPCA first.** Use the official MATLAB file, source labels only, unlabeled target features, and the same downstream classifier/evaluation split as the other methods. Keep target labels outside the adaptation function. Describe it as a 2023 generic linear UDA method evaluated on CPAR.
2. **Add ICSC only if you can make a clean label-blind wrapper.** Move target scoring out of `ICSC.m`, map labels to contiguous `1:C`, and report it as an additional method whose original tests were on image domains.
3. **Keep STL as the task-specific reference and CORAL/TCA/GFK as established shallow controls.** Do not infer a new method's expected performance from another dataset or paper.
4. **Do not add the current SOT Python implementation to the MATLAB baseline table as-is.** Its code language and published transfer protocol do not match the current requirement.
5. **Pre-register the experiment settings:** fixed source/target position pairs; a clear transductive or inductive target protocol; hyperparameters chosen without target test labels; repeated seeds or fixed published splits; accuracy and macro-F1; and uncertainty across task pairs/runs.

## Bibliography

- **[S01]** J. Wang, Y. Chen, L. Hu, X. Peng, and P. S. Yu. “Cross-position Activity Recognition with Stratified Transfer Learning.” *arXiv preprint* arXiv:1806.09776 (2018). https://arxiv.org/abs/1806.09776
- **[S02]** Project data description and MATLAB experiment: repository `readme.txt` and `demo.m` (local files). `readme.txt` documents the 81 features per body part; `demo.m` extracts position-specific source/target matrices.
- **[S03]** E. M. Mirkes, J. Bac, A. Fouché, S. V. Stasenko, A. Zinovyev, and A. N. Gorban. “Domain Adaptation Principal Component Analysis: Base Linear Method for Learning with Out-of-Distribution Data.” *Entropy* 25(1), 33 (2023). https://doi.org/10.3390/e25010033
- **[S04]** Mirkes et al., DAPCA author repository. https://github.com/Mirkes/DAPCA
- **[S05]** Mirkes et al., `DAPCA.m` source. https://github.com/Mirkes/DAPCA/blob/main/DAPCA.m
- **[S06]** S. Teng, Z. Zheng, N. Wu, L. Fei, and W. Zhang. “Domain Adaptation via Incremental Confidence Samples into Classification.” *International Journal of Intelligent Systems* 37(1), 365–385 (2022). https://doi.org/10.1002/int.22629
- **[S07]** Zheng et al., ICSC MATLAB repository. https://github.com/zzf495/ICSC
- **[S08]** Zheng et al., `ICSC.m` source. https://github.com/zzf495/ICSC/blob/master/ICSC.m
- **[S09]** Q. Xu, X. Wei, R. Bai, S. Li, and Z. Meng. “Integration of deep adaptation transfer learning and online sequential extreme learning machine for cross-person and cross-position activity recognition.” *Expert Systems with Applications* 212, 118807 (2023). https://doi.org/10.1016/j.eswa.2022.118807
- **[S10]** Y. Chang, A. Mathur, A. Isopoussu, J. Song, and F. Kawsar. “A Systematic Study of Unsupervised Domain Adaptation for Robust Human-Activity Recognition.” *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies* 4(1), Article 39 (2020). https://doi.org/10.1145/3380985
- **[S11]** P. Alinia, A. Arefeen, Z. Esna Ashari, S. I. Mirzadeh, and H. Ghasemzadeh. “Model-Agnostic Structural Transfer Learning for Cross-Domain Autonomous Activity Recognition.” *Sensors* 23(14), 6337 (2023). https://doi.org/10.3390/s23146337
- **[S12]** R. Hu, L. Chen, S. Miao, and X. Tang. “SWL-Adapt: An Unsupervised Domain Adaptation Model with Sample Weight Learning for Cross-User Wearable Human Activity Recognition.” *Proceedings of the AAAI Conference on Artificial Intelligence* 37(5), 6012–6020 (2023). https://doi.org/10.1609/aaai.v37i5.25743
- **[S13]** SWL-Adapt official code repository. https://github.com/Rxannro/SWL-Adapt
- **[S14]** H. Guan and M. Liu. “DomainATM: Domain adaptation toolbox for medical data analysis.” *NeuroImage* 268, 119863 (2023). https://doi.org/10.1016/j.neuroimage.2022.119863
- **[S15]** Guan and Liu, DomainATM MATLAB toolbox. https://github.com/guanharry/domainATM
- **[S16]** W. Lu, Y. Chen, J. Wang, and X. Qin. “Cross-domain Activity Recognition via Substructural Optimal Transport.” *Neurocomputing* 454, 134–147 (2021). https://arxiv.org/abs/2102.03353
- **[S17]** User's local SOT implementation, `sot/main.py` and `sot/SOT.py` (Python source).

## Methodology Appendix

I used standard research mode: define the target protocol from the user's repository, search 2020–2026 papers and code, inspect primary papers and author/official repositories, then apply a strict screening rule. A method qualified for the shortlist only when a public `.m` implementation was verifiable and its source/target data assumptions could plausibly consume the project's position-wise feature matrices. The screen prioritized paper/repository primary sources over surveys or secondary summaries.

The project scope was inferred from `readme.txt`, `demo.m`, and the original STL paper. Search checks covered cross-position HAR methods, MATLAB code, DAPCA, ICSC, ActiLabel, the 2023 cross-position deep method, DomainATM, SWL-Adapt, and SOT. Findings marked “inference” identify compatibility judgments rather than reported experimental evidence. No code was edited, no method was run, and no accuracy claim was made.
