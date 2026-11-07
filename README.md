# Sequential quote learning on Qwen2.5-7B-Instruct

**Author: Victoria Cendagorta Villalba. November 2026.**

A small, runnable continual fine-tuning pilot: learn Spanish quotation nesting at level 2, then level 3, and measure retention after each stage. This is evidence for a proposal, **not a solution or benchmark result for the client's book-learning task**.

## Main result

One LoRA adapter is trained sequentially on N2 and then N3; it is not reset or switched by task. On development pairs, N2 exact match is **0/7 before training → 6/7 after N2 → 7/7 after N3**. N3 is **0/12 → 2/12 → 6/12**. Development no-quote controls regress **5/5 → 4/5 → 4/5**. Counts below refer to unique input/target pairs, not duplicated rows.

| Development task | Base | After N2 | After N3 |
|---|---:|---:|---:|
| No-quote controls | 5/5 | 4/5 | 4/5 |
| N1 | 1/2 | 1/2 | 2/2 |
| N2 | 0/7 | 6/7 | 7/7 |
| N3 | 0/12 | 2/12 | 6/12 |

On training pairs, N2 reaches **17/17** after N2 and remains **17/17** after N3; N3 reaches **22/22** after N3. Retention of learned N2 is therefore observed on both training and development pairs. These are tiny, related tasks with replay; no causal claim is made for layer selection alone.

## New manual evaluation, frozen adapter

After training, the author wrote 12 new texts in four related families, three depths per family. Neither model receives demonstrations in its evaluation prompt. No further training took place.

| Task | Base | Final adapter |
|---|---:|---:|
| N1 | 2/4 | 3/4 |
| N2 | 0/4 | 2/4 |
| N3 | 0/4 | 2/4 |
| Total exact | 2/12 | 7/12 |

Six cases improve and one regresses. The final adapter preserves non-quote content in 12/12 cases, versus 11/12 for the base. It still adds, deletes or misassigns delimiters in some cases. This evaluation has four families, not twelve independent draws; it has no additional identity controls. These new references are never used in training.

## Exact recipe

- Base: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- LoRA on attention `q_proj`, `k_proj`, `v_proj`, `o_proj`, only layer indices **8–19** (human blocks 9–20).
- Rank 16, alpha 32, dropout 0.0, 4,325,376 trainable parameters. Base parameters, embeddings and output head remain frozen.
- Two stages of 40 optimizer steps; microbatch 1, accumulation 4; AdamW, learning rate 2e-4, cosine decay, 5% warmup, zero weight decay, gradient clipping 1.
- Replay among quote samples: stage N2 = 30% N1 / 70% N2; stage N3 = 20% N1 / 30% N2 / 50% N3. No-quote controls are 10% of all sampled records. Integer rounding is recorded in the sampling logs.
- Same adapter continues across stages; optimizer is reinitialized per stage. Seed 42. BF16, SDPA, gradient checkpointing; assistant-response-only loss. No truncation allowed.
- Input preprocessing replaces only quote delimiters with straight double quotes. All other content is retained. Exact match penalizes any change to words, punctuation or spaces.
- Evaluation uses the same fixed system instruction, no demonstrations, greedy decoding, 1024 maximum new tokens, no postprocessing repairs.

The recorded full training worker, including model loading and all three evaluations, took **285.57 seconds** on one **RTX A6000 48 GB**; the outer runner took **288.83 seconds**. Peak allocated GPU memory was **14.70 GB**. This tiny-run cost cannot be extrapolated to full books. The new manual evaluation took **50.16 seconds** including its wrapper.

## Reproduce

Use Python 3.11 and a CUDA GPU. The recorded stack is PyTorch 2.8.0+cu128, Transformers 5.22.1, PEFT 0.21.2, Accelerate 1.15.0, Safetensors 0.8.0, Hugging Face Hub 2.4.0. Start from a compatible PyTorch/CUDA image; the base weights are downloaded from the pinned official model revision. No API key is required for this public model; a Hugging Face token is optional for rate limits.

```bash
python3 -m pip install -r requirements.txt
# CPU-only: verify hashes, dataset isolation and recompute all recorded scores.
python3 reproduce.py verify
# GPU: compare the supplied final adapter and the base on the 12 new texts.
python3 reproduce.py evaluate
# GPU: repeat baseline → N2 → N3 and all development evaluations.
python3 reproduce.py train
```

`train` does not overwrite the supplied frozen checkpoint. It writes a timestamped run under `training/` and packs its results. `evaluate` uses the original frozen adapter, not a newly retrained one. Training plus a new manual evaluation of its resulting checkpoint can be run with:

```bash
python3 reproduce.py full
```

The `full` command uses the newly trained N3 adapter in a separate working directory and preserves the original evaluation references. Exact numeric identity of retrained GPU weights is not promised across hardware or software stacks. Dataset construction and sampling are seeded and recorded.

## Files

- `training/`: exact source, configuration and small dataset used for the recorded sequential run.
- `evidence/training/`: recorded references, predictions, scores, sampled IDs, losses, timing and environment.
- `manual_eval/`: new author-written references, deterministic neutralization, scorer, runner and frozen final adapter.
- `evidence/manual/`: recorded base/final predictions and per-case comparisons.
- `checkpoints/adapter_N2/`: intermediate adapter; final adapter is in `manual_eval/adapter_congelado/`.
- `REPORT.md`: interpretation and known failures; `PROVENANCE.md`: data and tooling provenance.
- `SHA256SUMS.json`: frozen hashes; `reproduce.py verify` recomputes the scores without a GPU.

## Limits and relevance to the book task

Only one training seed was run. Development families were held out from training but had been used for earlier diagnostics, so they are development data, not a pristine final test. New manual cases are fresh texts, with some structural resemblance to training. The pilot starts from the official base, **not the client's book-1 M1**, and trains N2 then N3; it does not demonstrate a newly trained N1 stage.

**No GSM8K, IFEval, MMLU or honesty panels were run. No claim is made that the client's thresholds, three-seed requirements or book-transfer outcome have been achieved.** Replay and restricted LoRA are tested jointly, without an ablation or comparison against KL-to-M1. Preserving frozen base weights does not guarantee preservation of model behavior.

The artifact demonstrates a reproducible sequential-training and forgetting-measurement workflow on a 7B model. Testing its transfer to close-topic books requires the client's M1, source units, answer setup and development feedback.
