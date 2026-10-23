# Loss harness

Nano GPT-2, random initialization, Hugging Face `GPT2LMHeadModel` with the GPT-2 tokenizer. Width 64, 2 layers, 4 heads, vocabulary 50,257. Dropout is off. Numbers below are from one CPU run of `loss_harness.ipynb`.

## Part 1

| # | Quantity | Value |
|---|----------|------:|
| 1 | Contributing tokens before the padding mask | 12 |
| 2 | Contributing tokens after the padding mask | 7 |
| 3 | Packed loss before the boundary mask | 0.7305 |
| 4 | Packed loss after the boundary mask | 0.1112 |
| 5 | Untrained perplexity | 53,317.8 |
| 6 | Tied parameter count | 3,332,928 |
| 7 | Untied parameter count | 6,549,376 |

Vocabulary size is 50,257. `log(50257) = 10.8249` nats. The untrained loss on 69 tokens was 10.884, so perplexity is `exp(10.884) = 53317.8`, which is 1.061 times the vocabulary size.

Padding: `"Hello."` beside `"The capital of France is Paris."` is 2 rows of width 7, so the shift offers 12 targets. GPT-2 has no pad token (`pad_token is None`); the id used for padding is the end-of-text id 50256. The mask is `attention_mask`, not the id. After `-100` on padded targets, 7 tokens remain: `Hello → .`, and the six next-token pairs in the longer sentence.

Boundary: `"The capital of France is Paris."` and `"Taj Mahal stands in Agra."` are packed with `<|endoftext|>` between them. Each document is learned on its own first, where that end-of-text token is the last position and the shift never scores a successor. In the packed row the same token's target is `T`, the first piece of "Taj". That one term is 10.6387 nats. The mean is 0.7305 with it and 0.1112 without it, and `(0.7305 × 17 − 10.6387) / 16 = 0.1112`. On the untrained model the same mask moves the mean from 10.8447 to 10.8517, because every target is already near `log(V)`.

Tied weights: `_tied_weights_keys` maps `lm_head.weight` onto `transformer.wte.weight`, and with `tie_word_embeddings=True` they are one storage. The untied model adds `50257 × 64 = 3,216,448` parameters. Counting the tied matrix twice reproduces the untied total, 6,549,376.

Peak memory of the loss, 2048 token states against this head, chunk length 64. The two losses match (10.835638).

| | Peak |
|--|--:|
| Ordinary cross-entropy | 785.27 MB |
| Chunked cross-entropy | 25.06 MB |
| Ratio | 31.34× |

The ordinary peak is the `(2048, 50257)` score matrix plus the log-softmax beside it. The chunked path keeps one slice of 64 rows at a time.

## Part 2

A second head, an independent copy of the tied weights, predicts token `t+2`. Losses are means over valid targets. There are 150 `t+1` targets and 138 `t+2` targets.

| Step | t+1 | t+2 | Sum |
|-----:|----:|----:|----:|
| 0 | 10.8290 | 10.8237 | 21.6528 |
| 40 | 5.7493 | 5.6092 | 11.3586 |

Both start on `log(50257)`. Over 40 steps they fall together, by 5.08 and 5.21 nats. On the 138 positions that have both targets, the step-40 losses are 5.7853 (`t+1`) and 5.6092 (`t+2`). The 12 tail positions only the first head scores sit at 5.3359. Both heads began as the same random matrix, and on a fixed batch each shift is a lookup from the same hidden state, so the nearer token has no head start. The curves stay together.

## Part 3

`GPT2LMHeadModel.forward` documents that labels are shifted inside the model, and that `labels = input_ids` is the right argument. In this transformers, `ForCausalLMLoss` does that by padding `-100` on the right of the labels and taking `labels[..., 1:]`. It does not slice the logits. That pairs the same tokens as `logits[:, :-1]` against `tokens[:, 1:]`.

On the probe sentence the hand-sliced loss and `model(..., labels=input_ids).loss` are the same number, 9.6348.

A second shift is invisible in an untrained mean, because every class is about equally wrong. It is visible in the strings, and in a row of logits spiked at the true next token:

| | Loss |
|--|--:|
| One shift, spiked logits | 0.0 |
| Library loss on `labels = input_ids` | 0.0 |
| Labels already shifted, then shifted again | 50.0 |

Position 0's input is `The`. The correct target is ` capital`. After the extra shift the target is ` of`.
