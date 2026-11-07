# Data and tooling provenance

Author and corpus provider: Victoria Cendagorta Villalba.

The sequential dataset is the exact `pares.jsonl` and `controles.jsonl` stored by the successful training run. It contains 102 quoted records (72 unique input/target pairs) from six manually authored template families, plus 11 editorial identity controls (11 unique). The quoted split is by pattern: P2/P3/P4/P6 train, P5/P7 development. Control documents are separated between train and development. Repeated rows are retained; scores explicitly distinguish raw records from unique pairs.

Human templates were written by the author. Lexical variations were generated locally with the permitted Qwen2.5-7B-Instruct teacher. Deterministic template assembly, quote substitution and agreement cleanup were used. The P1 family was excluded because its provenance mixed assistant-provided example material; the validator refuses it. Earlier GPT-written synthetic exploration datasets and failed later generation batches are not included as training data.

The 12 new evaluation texts were written by the author after training. Their final complete version is copied in `manual_eval/textos_autora.json`, including intentionally retained names, punctuation and colloquial language. Only paragraph separators and trailing presentation whitespace are omitted. The input transformation changes the six quote characters to ASCII double quotes and nothing else. References were frozen and checked before any predictions on those texts were obtained. They are never supplied as training examples or inference demonstrations.

Closed-model assistance was used to write code, inspect errors and produce documentation. The supplied provenance describes the final text route; it does not claim independent third-party certification of all upstream corpus rights. No client book text, client hidden panel or general benchmark item is included. The author's full editorial corpus is not distributed.

Code license: MIT (`LICENSE`). The small datasets and author-owned documentation are made available by the author for reproduction under CC BY 4.0 (`DATA_LICENSE.md`), to the extent of the author's rights; underlying third-party rights, if any, are not reassigned. Qwen base weights are not included; obtain them from the official Hugging Face repository under its own license. Checkpoint use remains subject to applicable base-model terms. Dependencies retain their respective licenses.
