# Reading list, in priority order

Ordered by **when you need each paper**, not by how important it is in general. Tier 1 unblocks the scoping gate (counting trials → power analysis → dataset and human target → pre-registration). Tier 2 is split by owner and lines up with each person's tasks and Related Work sections. Read Tier 3 only if the matching method turns out to be needed.

**B** = Borjan, **A** = Andrej.

---

## Tier 1: both of us, this week (core of the topic)

| # | Paper | Why read it now | Focus on |
|---|---|---|---|
| 1 | **Ellsberg (1961)**, Risk, Ambiguity, and the Savage Axioms. *QJE* 75:643–669. [doi](https://doi.org/10.2307/1884324) | The definition of the whole project. The two-urn and single-urn (three-colour) set-ups are the templates for our synthetic probe set. | The two urn examples; why they violate Savage. |
| 2 | **Zhu, Yan & Griffiths (2025)**, Steering risk preferences in LLMs by aligning behavioral and neural representations. [arXiv:2505.11615](https://arxiv.org/abs/2505.11615) | The closest prior work: same domain, same kind of method, and it beats the diff-of-means pipeline we were going to use. Defines our comparison method. | The alignment pipeline (MCMC elicitation → Lasso), their contrastive baseline, and the evaluation metrics. |
| 3 | **Arditi et al. (2024)**, Refusal in language models is mediated by a single direction. *NeurIPS*. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) | The method recipe we follow: diff-of-means → directional ablation → capability check. Most of our pipeline is this paper applied to ambiguity. | Methods sections: direction selection, ablation, and the evaluation. |
| 4 | **Peterson et al. (2021)**, Using large-scale experiments and machine learning to discover theories of human decision-making. *Science* 372:1209–1214. [doi](https://www.science.org/doi/10.1126/science.abe2629) | Source of choices13k, and our evidence for the gap: their preprocessing removes the ambiguous trials. Needed for counting the ambiguity trials. | The supplement on data and preprocessing: how ambiguity is encoded and what gets dropped. |
| 5 | **Erev, Ert, Plonsky, Cohen & Cohen (2017)**, From anomalies to forecasts. *Psych Review* 124:369–409. [doi](https://doi.org/10.1037/rev0000062) + the **CPC18** release docs. [zenodo](https://zenodo.org/records/2571510) | The design behind CPC15/18, including the deliberate ambiguity manipulation and the feedback blocks. Needed for counting trials, the power analysis and choosing the dataset. | How the ambiguity parameter is coded, and the block structure (no-feedback vs. feedback). |

---

## Tier 2: split by workstream (weeks 1–3)

### Borjan: extraction, steering and LLM cognitive modeling (Related Work: LLM cognitive modeling; steering and extraction)

| # | Paper | Why |
|---|---|---|
| B1 | **Belrose (2023)**, Diff-in-means concept editing is worst-case optimal. [blog](https://blog.eleuther.ai/diff-in-means/) | Short. The theoretical reason we edit with diff-of-means instead of a probe weight vector. |
| B2 | **Panickssery (Rimsky) et al. (2024)**, Steering Llama 2 via contrastive activation addition. *ACL*. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681) | Where our coefficient-sweep protocol comes from. |
| B3 | **Wollschläger et al. (2025)**, The geometry of refusal in LLMs. *ICML*. [arXiv:2502.17420](https://arxiv.org/abs/2502.17420) | Why we run a rank sweep before claiming a single direction. |
| B4 | **Tan et al. (2024)**, Analysing the generalisation and reliability of steering vectors. *NeurIPS*. [arXiv:2407.12404](https://arxiv.org/abs/2407.12404) + **Braun et al. (2025)**, Understanding the (un)reliability of steering vectors. [arXiv:2505.22637](https://arxiv.org/abs/2505.22637) | Why in-distribution and out-of-distribution steering are reported separately. Needed for the steering task and the Limitations. |
| B5 | **Binz et al. (2025)**, Centaur. *Nature* 644:1002–1009. [link](https://www.nature.com/articles/s41586-025-09215-4) + **Xie & Zhu (2025)**, the shortcut critique. [doi](https://doi.org/10.31234/osf.io/u7z4t_v1) | The main LLM-as-cognitive-model claim, and the critique of it. Both are needed for the Related Work section. Centaur also claims its internals become more human-like, which our probes can test rather than assume. |
| B6 | **Zhu, Xie, Arumugam, Wilson & Griffiths (2025)**, RL to train LLMs to explain human decisions. [arXiv:2505.11614](https://arxiv.org/abs/2505.11614) | "The RL/SFT paper" the Related Work task mentions. Skim it: it's another risky-choice-only paper. |
| B7 | **Turner et al. (2023)**, Activation addition [arXiv:2308.10248](https://arxiv.org/abs/2308.10248) and **Zou et al. (2023)**, Representation engineering [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) | Skim. The origins of steering by addition and of contrastive reading vectors, cited as baselines. |

### Andrej: human ambiguity attitudes, probing and statistics (Related Work: Ellsberg and behavioral economics; human effect sizes)

| # | Paper | Why |
|---|---|---|
| A1 | **Voorhoeve, Binmore, Stefansson & Stewart (2016)**, Ambiguity attitudes, framing, and consistency. *Theory and Decision* 81:313–337. [link](https://link.springer.com/article/10.1007/s11238-016-9544-1) | The paper behind our human-target link. Read it before fixing the human target, and trace where the 66% / 27% numbers come from (check its literature review). It also finds aversion **survives** loss framing, which matters for the framing task. |
| A2 | **Trautmann & van de Kuilen (2015)**, Ambiguity attitudes (*Wiley Blackwell Handbook of Judgment and Decision Making*) + **Epstein & Schneider (2010)**, Ambiguity and asset markets (*Annual Review of Financial Economics*) | The standard ways to measure ambiguity attitudes (matching probabilities, ambiguity premia, source functions). We adopt one rather than inventing our own. Trautmann & van de Kuilen is also the best overview of effect sizes across designs. (The original plan lists this survey as "Trautmann & Wakker"; confirm which one is meant.) |
| A3 | **Hewitt & Liang (2019)**, Designing and interpreting probes with control tasks. *EMNLP*. [arXiv:1909.03368](https://arxiv.org/abs/1909.03368) | Without a control task, the H3B probe result can be dismissed as probe capacity. Needed for the probe tasks. |
| A4 | **Marks & Tegmark (2024)**, The geometry of truth. *COLM*. [arXiv:2310.06824](https://arxiv.org/abs/2310.06824) | A template for probing a **scalar** latent across layers. Our H3B probe reads a scalar too (stated probability). |
| A5 | **Kahneman & Tversky (1979)**, Prospect theory. *Econometrica*. [doi](https://doi.org/10.2307/1914185) | We must name a reference model (EV, CPT or BEAST) for the "pure-EV prediction". |
| A6 | **Binz et al. (2025)**, Centaur (same as B5) | Needed for the Centaur baseline: the Psych-101 task list for the contamination check. |
| A7 | **Alain & Bengio (2016)**, Linear classifier probes. [arXiv:1610.01644](https://arxiv.org/abs/1610.01644) | Skim. The original probing baseline. |

### Cross-read: each reads one paper from the other's list

- **Both:** **Park, Choe & Veitch (2024)**, The linear representation hypothesis and the geometry of LLMs. *ICML*. [arXiv:2311.03658](https://arxiv.org/abs/2311.03658). It explains why a probe direction (Andrej) and a steering direction (Borjan) don't have to match, which is exactly the issue when we compare our directions.
- **Borjan:** read A1 (Voorhoeve et al.) so you know where the human target comes from.
- **Andrej:** read B1 (Belrose) so you know why ablation uses diff-of-means.

---

## Tier 3: only if needed, or when writing

| Paper | Read it when |
|---|---|
| **Thomas et al. (2024)**, Modelling dataset bias in machine-learned theories of economic decision-making. *Nat Hum Behav*. [link](https://www.nature.com/articles/s41562-023-01784-6) | Writing the Limitations: our results are about choices13k/CPC18, not about human decision-making in general. |
| **Geiger, Wu et al. (2023)**, Distributed alignment search. [arXiv:2303.02536](https://arxiv.org/abs/2303.02536) | The ablation is clearly causal but diff-of-means finds nothing. |
| **Lieberum et al. (2024)**, Gemma Scope [arXiv:2408.05147](https://arxiv.org/abs/2408.05147); **Bricken et al. (2023)**, Towards monosemanticity [link](https://transformer-circuits.pub/2023/monosemantic-features/index.html); **Cunningham et al. (2023)**, Sparse autoencoders find highly interpretable features [arXiv:2309.08600](https://arxiv.org/abs/2309.08600); **Templeton et al. (2024)**, Scaling monosemanticity [link](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html); **Wu et al. (2025)**, AxBench [arXiv:2501.17148](https://arxiv.org/abs/2501.17148) | We decide to interpret the direction with SAE features (Gemma-2 only). AxBench answers "why didn't you steer with SAEs?". |
| **Zhang & Nanda (2024)**, Best practices of activation patching [arXiv:2309.16042](https://arxiv.org/abs/2309.16042); **Meng et al. (2022)**, ROME [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) | We add a layer × position patching heatmap. |
| **Li et al. (2023)**, Inference-time intervention [arXiv:2306.03341](https://arxiv.org/abs/2306.03341) | Residual-stream steering hurts fluency. |

Papers in the original combined plan that are **not** on this list (Kirk et al., DeepSeekMath/GRPO, emergent misalignment, persona features/vectors, crosscoders, path patching, ACDC, IOI, all the personality papers) belong to the other two projects.
