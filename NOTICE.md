# PHSMLP - Pooled Head MLP: copyright, attribution and provenance

Required Notice: Copyright (c) 2026 Rickey1990, original project author, rickeyernest1990@gmail.com.

## Original project software

The original PHSMLP software in this repository is offered under the [PolyForm Noncommercial License 1.0.0](LICENSE), following the licensing approach of the author's [Gated-Attention-Projection (GAP20) project](https://github.com/rickey1990/Gated-Attention-Projection). For commercial permissions, attribution questions or corrections, contact rickeyernest1990@gmail.com. This is a separate pooled-head/shared-MLP architecture study, not the GAP20 fused-gate block or a reproduction of its WikiText-103 experiment.

The report PDF carries its own copyright notice. The software licence does not assert exclusive rights over general equations, abstract methods, attention, MLP weight sharing or independently written implementations. Credit the original experimental provenance when reusing these results or adapting the project code. Retain this notice and the licence when redistributing licensed software, including the source embedded in the notebook and the copy inside the audit archive.

## Third-party data and dependencies

The training source was **FineWeb-Edu**, published by Anton Lozhkov, Loubna Ben Allal, Leandro von Werra and Thomas Wolf (2024), DOI [10.57967/hf/2497](https://doi.org/10.57967/hf/2497). See the [dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/) and [Common Crawl terms of use](https://commoncrawl.org/terms-of-use). FineWeb-Edu is provided under ODC-By 1.0 and is subject to Common Crawl's terms; rights in individual underlying web content may differ. The project software licence does not replace those third-party terms or grant rights in third-party content.

The repository does not redistribute the raw training corpus, tokenised corpus or trained weights. The original audit contains experimental measurements, synthetic probe records, a trained tokenizer and its vocabulary, dataset fingerprints and the experiment implementation; these are retained as research provenance. Dataset-derived artifacts remain subject to applicable third-party rights. PyTorch, NumPy, SentencePiece, Hugging Face datasets, PyArrow and other dependencies retain their own licences; they are not relicensed here.

## Experimental provenance

The report analyses the supplied completed Kaggle run: 500M scored targets per model, four architectures, one seed (20261109), on a fixed local FineWeb-Edu sample-10BT shard. `research_results.zip` preserves the supplied audit unmodified. The notebook embeds the same experiment source (ignoring the final newline). The public files permit inspection, audit and retraining with external data and suitable compute; they do not supply pretrained checkpoints or claim independent reproduction, multi-seed confirmation or superiority over the standard Transformer.
