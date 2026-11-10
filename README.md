# PHSMLP - Pooled Head MLP

**500M-token Transformer study**

PHSMLP is a controlled four-model experiment investigating whether **one shared feed-forward network (MLP)** and an additional learned attention-head pooling path can provide a useful parameter-efficient Transformer. Each model processed 500 million scored FineWeb-Edu next-token targets, for 2 billion total. All four completed training without recorded numerical collapse.

**[Read the eight-page research report (PDF)](PHSMLP_500M_Research_Report.pdf)** · **[Download the Kaggle notebook](PHSMLP_500M_Kaggle.ipynb)** · **[Download the original research audit](research_results.zip)**

## Main findings

| Final held-out result | Standard | Standard + pool | Shared MLP | Shared + pool |
|---|---:|---:|---:|---:|
| Trainable parameters | 444,992 | 451,331 | 378,816 | 385,155 |
| Test perplexity (lower is better) | **88.75** | 90.60 | 105.98 | 105.66 |
| Next-token accuracy | **23.95%** | 23.50% | 21.18% | 21.03% |

The shared pooled model uses **13.4% fewer total parameters** than standard. It learns and remains stable throughout the 500M-token run. Its learned pooling path contributes to prediction: disabling the path in a trained model increases perplexity from 100.02 to 181.20 on the same 8,192-position ablation sample. This establishes learned dependence on the path, not that pooling is inherently better than training an unpooled model.

The standard Transformer retains the strongest language prediction and copying results. Shared + pool's final perplexity is about 19% higher than standard, and the validation gap widens over training. The small difference between shared + pool and plain sharing is inconclusive in paired window statistics. All models struggle with zero-shot entity binding, corrections and multihop probes; no model produced a correct unrestricted single-token answer in the final suite. Pooling also reduced measured training throughput.

**Interpretation:** successful feasibility study, not demonstrated architectural superiority. Further training may improve absolute quality, but catching up is an open question rather than an established outcome. An equal-parameter wider shared MLP, pass-specific adaptations, matched-compute comparisons and independent seeds are the next experiments. This is a separate study from [GAP20](https://github.com/rickey1990/Gated-Attention-Projection).

## Architecture

Three processing passes; four independent attention heads per pass; width 64; FFN hidden width 256; causal context 512; 4,096-piece SentencePiece BPE vocabulary with tied input/output token embeddings. The four-way design crosses independent/shared MLPs with ordinary/pooled attention.

The pooled path is additive, not a replacement for standard attention:

```text
u = LN(x)
a_h = causal_attention(Q_h(u), K_h(u), V_h(u))
b = W_A concat(a_1, a_2, a_3, a_4) + b_A
w_h = softmax_h((W_q u) dot a_h / sqrt(16))
s = sum_h w_h a_h
A(u) = b + sigmoid(l) * (W_p s + b_p)
z = x + A(LN_a(x))
y = z + F(LN_f(z))
F(v) = W_2 GELU(W_1 v + b_1) + b_2
```

The unpooled variants use `A(u) = b`. The shared variants reuse the same `F` at all three passes; attention and normalisation remain pass-specific. Attention heads and pooling are differentiable and jointly trained. All heads run in parallel; the three processing passes are sequential. Sharing saves weights, not the repeated FFN evaluations.

## Six repository files

| File | Purpose |
|---|---|
| `README.md` | Project overview, results and instructions |
| `PHSMLP_500M_Research_Report.pdf` | Research report, mathematics, plots, evaluations and limitations |
| `PHSMLP_500M_Kaggle.ipynb` | Self-contained implementation, preparation, training, health checks, checkpointing and evaluation |
| `research_results.zip` | Original compact audit with raw predictions, metrics, manifests, tokenizer and exact Python implementation |
| `LICENSE` | PolyForm Noncommercial License 1.0.0 for original project software |
| `NOTICE.md` | Copyright, data attribution, provenance and licensing boundaries |

## Run on Kaggle

1. Download the [FineWeb-Edu Parquet shard](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/20b2c9f0d743a587e1be1cc836d0ea084fd662d4/sample/10BT/000_00000.parquet) separately. Create a Kaggle Dataset containing `000_00000.parquet` and attach it to a notebook made from the included `.ipynb` file. Enable GPU. Internet is needed only if dependencies must be installed.
2. Run the cells in order. The notebook writes `research500.py`, validates its syntax, runs the optional small smoke test, prepares the corpus from the local Parquet and trains the four models on the same frozen target schedule. The full run is GPU-only. Default seed is 20261109; all four configurations use the same data order and budget.
3. Retain the experiment folder for resumption: token cache, tokenizer, latest checkpoints including optimiser/scaler state, logs and configuration. Attach saved outputs and set `RESUME_INPUT_ROOT` in the notebook before resuming. The compact audit ZIP alone cannot resume training.
4. When all four models complete, download `RESEARCH_RESULTS_UNDER_350MB.zip` from Kaggle output. Compare its manifest and final completion file to the included original audit.

Training checkpoints and health checks are recorded every 1M targets; regular probes every 10M, expanded probes and ablations every 25M, and final held-out evaluation at 500M. The notebook retains latest resumable checkpoints plus periodic archived snapshots rather than placing every checkpoint in the compact results ZIP.

### Audit the existing results

Unzip `research_results.zip`. The original `research500.py` inside is the full training/evaluation implementation; the notebook embeds the same source, apart from its final newline. For inspection without retraining, start with:

- `reports/final_complete.json`, `reports/final_results.json` and `model_specs.json` for final completion, performance and parameter budgets.
- `metrics/health.csv` for all 2,000 milestone rows; `metrics/probe_milestones.csv` for probe history.
- `probes/test/final_probe_rows.csv`, `probes/test/copying.csv` and `probes/pool_ablations.csv` for detailed final tests. For main-suite comparisons, select normal pooling mode and the first occurrence per model/probe ID to exclude duplicated mini-ablation rows. There are 1,408 matched main questions per model.
- `reports/paired_final_bootstrap.json` and `reports/paired_test_nll_bootstrap.json` for paired sample uncertainty, not variation across training seeds.
- `data/manifest.json`, `data/source_revision.json`, `experiment_config.json` and `experiment_manifest.json` for the source, tokenizer/token-file hashes, exact settings and environment.

The recorded local shard SHA-256 is `c4c01ab0833ecff9bd5bf5924e82006366b68ef41e8632dafde69ef5a3f97322`. Verify the actual file before claiming a matched reproduction. The original audit SHA-256 is `dbd357800a1e7d1f5bb9b633f3e06140cee05275abd08b2254c96920a6a5411d`.

The repository does **not** include raw text, the 500M token cache or model checkpoints. Reviewing measurements is possible immediately; independently reproducing the experiment requires external data, software, compute and retraining. Results may vary across software/hardware and seeds. The final language evaluation uses 100,000 scored held-out targets, and synthetic probes are zero-shot, not task-trained reasoning benchmarks. See the PDF for caveats and proposed next steps.

## Licence and attribution

Original project software: [PolyForm Noncommercial License 1.0.0](LICENSE). Required Notice: Copyright (c) 2026 Rickey1990, original project author, rickeyernest1990@gmail.com. The PDF carries its own copyright notice. See [NOTICE.md](NOTICE.md) for scope, third-party rights and commercial contact.

FineWeb-Edu: Lozhkov, Ben Allal, von Werra and Wolf (2024), [dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [DOI](https://doi.org/10.57967/hf/2497). Its data licence is ODC-By 1.0, subject also to Common Crawl terms. Third-party data and software licences remain separate from the project's software licence.
