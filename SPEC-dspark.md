# Inkling 2 Mini: DSpark drafter (mtp.safetensors) implementation spec

Research-only, 2026-11-22. Extends SPEC.md §9. Written in our own words, with no code copied.
**Sources used:** `config.json` `mtp_config`. Official and uncensored copies are identical, and neither repo has
python files. The header of `mtp.safetensors` (names, shapes and dtypes only; no values were read). The official
README, which never mentions the drafter. The uncensored card, which says the drafter is unchanged. The TM blog, the TM
model card and the Tinker Inkling page, which say nothing about it. The HF launch blog (huggingface.co/blog/inkling2mini)
has one useful sentence: "the draft reads the inputs to decoder visits 5, 23, and 41, plus the final normalized state
at index 48". The **DSpark paper** (arXiv 2607.05147, DeepSeek-AI/PKU), which gives the architecture family. From
Transformers: PR #51050, which adds the target model with **no drafter code**; open PR #49996 "[Feature] Add DSpark",
which is generic (DeepSeek Qwen3 drafters, not Inkling) and handles the "vanilla" and "rnn" heads but not Inkling's
`rnn_hidden`; the generic DFlash path on main (`DFlashTokenCandidateGenerator`, `MuseGlimmerAssistantModel`); and
`utils/output_capturing.py` for the hidden-state indexing. No SGLang, vLLM, llama.cpp, ingine, maya or basalt material
was used. A web-search summary mentioned some serving flags from those engines; I ignored it.
Fetched files are in `/tmp/dspark-spec/` (data only).

**No official source gives Inkling's drafter math.** The weight names and shapes, the config, the paper and the generic
HF DFlash/DSpark code together pin the overall structure. Everything Inkling-specific (`fuse_weight`, `dflare_norm`, the
`rnn_hidden` head and the ctx projections) is inferred. Tags: **[C]** = confirmed by a source, **[U]** = UNCONFIRMED,
with the most plausible reading given.

## 0. What DSpark is (paper §2.2, §3.1) [C]
- The drafter is a **parallel backbone** (DFlash: a few dense layers in which the block attends bidirectionally to
  itself and to K/V injected from the target's hidden states) plus a light **sequential head** that adds a bias that
  depends on the prefix.
- Input block: the anchor token x0 (the bonus token the target just emitted, which the target has not yet processed)
  followed by γ−1 mask embeddings. Output: **γ draft distributions**. Row 0, whose input is the anchor, predicts the
  token after the anchor. DSpark changed this from DFlash, which drops row 0.
- Draft k ~ p_k(v) ∝ exp(U_k(v) + B_k(x0, x<k, v)), sampled left to right. U_k = base logits from the backbone; B_k
  = the head's bias.
- The paper's RNN head: z_k = [s_{k-1}; W1[x_{k-1}]; h_k]. One linear maps z_k to (g, c, o), each r = 256.
  s_k = σ(g)⊙s_{k-1} + (1−σ(g))⊙tanh(c). B_k = W2ᵀ tanh(o). s_0 = 0.
- The paper also has a confidence head and a hardware-aware prefix scheduler. **Inkling ships neither**: no
  `confidence_head.*` tensors. Training: the target is frozen and the drafter **shares the target's embedding and LM
  head**; the loss is 0.1·CE + 0.9·TV distance to the target distribution. Evaluation: standard (lossless) speculative
  sampling at **temperature 1.0**.

## 1. Config (`mtp_config`) [C]
hidden 2304, 3 layers, 24 q heads / 8 kv heads × 128 (GQA 3:1), dense MLP 9216, rms eps 1e-6, rope_theta 1e6,
max_pos 1,048,576, no q/o bias, `local_layer_ids [0,2]` (sliding, window 1024; layer 1 = full attention),
`block_size 8`, `markov_rank 256`, `markov_head_type "rnn_hidden"`, `target_layer_ids [5,23,41,48]`, bf16.
There is no `vocab_size`, `mask_token_id` or `hidden_act` key, no embedding table and no lm_head in the file.

## 2. Tensors (all BF16; 317,897,872 params = 636 MB) [C names/shapes; role C/U as marked]
| tensor | shape | role |
|---|---|---|
| `fuse_weight` | [4,4] | mixes the 4 target taps into 4 context features [U §3] |
| `mask_embed` | [2304] | input vector for block rows 1..7. Stands in for DFlash's mask-token embedding [C by shape + paper] |
| `layers.{0,1,2}.input_layernorm.weight` | [2304] | pre-attention RMSNorm (gain) [C-ish, standard] |
| `layers.L.dflare_norm.weight` | [2304] | RMSNorm of this layer's fused context feature before `*_ctx` [U] |
| `layers.L.self_attn.q_proj.weight` | [3072,2304] | Q from the block hidden |
| `layers.L.self_attn.k_proj.weight`, `v_proj.weight` | [1024,2304] | K/V of the **block** rows |
| `layers.L.self_attn.k_proj_ctx.weight`, `v_proj_ctx.weight` | [1024,2304] | K/V of the **context** (target) rows |
| `layers.L.self_attn.q_norm.weight`, `k_norm.weight` | [128] | per-head RMSNorm on Q and on block K (before RoPE) |
| `layers.L.self_attn.k_norm_ctx.weight` | [128] | per-head RMSNorm on context K. Context V has no norm |
| `layers.L.self_attn.o_proj.weight` | [2304,3072] | attention output |
| `layers.L.post_attention_layernorm.weight` | [2304] | pre-MLP RMSNorm |
| `layers.L.mlp.{gate,up}_proj.weight` / `down_proj.weight` | [9216,2304] / [2304,9216] | SwiGLU, act = SiLU [U: no `hidden_act`; DFlash/Qwen3 default] |
| `norm.weight` | [2304] | final RMSNorm of the backbone output → h_k |
| `markov_head.markov_w1.weight` | [201024,256] | W1: low-rank embedding of the previous drafted token (rows = padded vocab) |
| `markov_head.rnn_proj.weight` / `.bias` | [768,5120] / [768] | one linear z→(g,c,o), 3×256. 5120 = 256+256+2304+2304 [U split §5] |
| `markov_head.hidden_out.weight` / `.bias` | [2304,256] / [2304] | maps tanh(o) (256) into hidden space: this replaces the paper's W2 (r→V) ("rnn_hidden") |
| `markov_head.dflare_norm.weight` | [2304] | RMSNorm of the head's own 2304-wide extra input [U §5] |
Shared from the target (not in the file): the embedding table (anchor row), the unembed (first 200058 rows) and
maybe the 30·tanh(x/30) softcap [U §6].

## 3. Inputs: target taps and fuse_weight
**Tap indexing [C for HF convention; mapping to Inkling ids U but strongly supported].** HF `output_hidden_states`
is a list of 49 entries for our 48 visits. `hs[0]` = the input of visit 0 (x0, the normed embedding). `hs[k]` = the
output of visit k−1 = the bf16 `h` passed into visit k. The last entry `hs[48]` is overwritten with the final
normalized state, `RMSNorm_w(AttnRes(output_query, Blocks, P); norm.weight)`, which is the vector fed to the unembed.
The HF blog's wording ("inputs to visits 5, 23, 41, plus the final normalized state at index 48") matches reading
`target_layer_ids` as **direct `hs` indices**. The generic HF DFlash rule (`hs[id+1]`) cannot apply here, because
index 49 does not exist.
- tap0 = hs[5] = input of visit 5 (layer 5, full attention) = output of layer 4.
- tap1 = hs[23] = input of visit 23 (pass 2, layer 11, full) = output of pass-2 layer 10.
- tap2 = hs[41] = input of visit 41 (pass 3, layer 17, full) = output of pass-3 layer 16.
- tap3 = hs[48] = final normed state (bf16).
- Visit numbering: 0..5 = layers 0..5; 6..17 / 18..29 / 30..41 = loop passes 1/2/3 (layers 6..17); 42..47 = layers
  18..23. None of the taps sits at a loop-restart visit (18, 30), so "input of k" equals "output of k−1".
- Taps 0..2 are **attn-res partial sums `h`** (the per-block partial), not a full residual stream, because that is
  what the HF layers take as input. Their scale differs from tap3. In our engine they are the `h` going into those
  visits (`hbuf`).
- **Fusion [U]:** there is no `fc` (DFlash normally has W_c ∈ R^{d×4d}), so the only mixer is the scalar `fuse_weight`
  [4,4]. Most plausible: 4 consumers (layers 0, 1, 2 and the markov head) × 4 taps, each consumer with its own
  `dflare_norm`:
  `ctx_c[t] = RMSNorm_w( Σ_j F[c,j] · tap_j[t] ; dflare_norm_c )`, with c = 0,1,2 → layers, c = 3 → head.
  Open points: row versus column orientation (try F and Fᵀ); whether each tap is RMS-normalized (no weight) before
  mixing (plausible, given the scale gap between partial sums and the normed final); and whether F is
  softmax-normalized per row. Mix and norm in fp32, output bf16.

## 4. Backbone: one draft round (anchor x0 at absolute position p)
Context = target taps for positions 0..p−1. These are all the tokens the target has already processed, accepted
drafts included.
```
X[0] = E(x0); X[1..7] = mask_embed            # E: target embed row. Raw or embed_norm'd? [U §6]
pos_blk = p .. p+7 ; pos_ctx[t] = t
for L in 0..2:
    a  = RMSNorm_w(X; input_layernorm_L)                   # [8,2304]
    Q  = rope(rmsnorm_head(a@Wq ᵀ -> [8,24,128]; q_norm_L), pos_blk)
    Kb = rope(rmsnorm_head(a@Wk ᵀ -> [8,8,128];  k_norm_L), pos_blk);  Vb = a@Wv ᵀ
    Kc = rope(rmsnorm_head(ctx_L@Wk_ctx ᵀ; k_norm_ctx_L), pos_ctx);    Vc = ctx_L@Wv_ctx ᵀ   # cached, §7
    keys = [Kc ; Kb], vals = [Vc ; Vb]; q head h uses kv head h//3; scale 1/sqrt(128); fp32 softmax
    mask: every block row sees ALL 8 block rows (bidirectional) and context rows t < p;
          sliding layers (0, 2) further require pos_q − pos_k < 1024 (boundary < vs <= [U])
    X  = X + (attn @ Wo ᵀ)
    m  = RMSNorm_w(X; post_attention_layernorm_L);  X = X + Wdown(silu(Wgate m) * (Wup m))
H = RMSNorm_w(X; norm)                         # h_0..h_7, bf16 [8,2304]; row k drafts position p+k+1
```
- RoPE: standard rotate-half (NeoX) with inv_freq = 1e6^(−2i/128) over the full 128 dims, applied after the per-head
  q/k norms. [C for HF DFlash; U for Inkling, but the most plausible reading.] **The target has no RoPE. The drafter
  does, and uses absolute token positions** (image tokens count). Nothing is clipped, softcapped or log-scaled, and
  there is no sink or rel-bias.
- Inkling's change from DFlash: the context has **its own** K/V projections and k-norm (`*_ctx`), and each layer has
  its own fused input. Plain DFlash shares W_K/W_V and uses one H_ctx for every layer. [C by tensor names]

## 5. Sequential "rnn_hidden" head (8 steps, left to right) [structure C from paper; Inkling details U]
```
s = 0 (256); prev = x0; e_ctx = ctx_3[p-1]     # head's fused feature at the last context row [U]
for k in 0..7:
    z  = concat(s, W1[prev], h_k, e_ctx)        # 256+256+2304+2304 = 5120; order [U] (paper order: s, W1, h)
    g, c, o = split(z @ rnn_proj.Wᵀ + rnn_proj.b, 256)   # order g,c,o as in the paper [U]
    s  = σ(g)⊙s + (1−σ(g))⊙tanh(c)              # paper uses s_{k-1} in z and o from z (not from the new s)
    b  = hidden_out(tanh(o))                     # [2304]
    logits_k = softcap30( unembed[:200058] @ (h_k + b) )   # = U_k + B_k with W2 := unembed∘hidden_out [U §6]
    q_k = transform(logits_k; temperature, top_k, top_p)   # same rules as spec.probs
    d_k = sample(q_k) (or argmax when greedy); prev = d_k
```
- Other readings of the 4th input (5120 − 2816 = 2304) and `markov_head.dflare_norm`: (b) the target embedding
  E[prev] in full 2304 width, normed by `dflare_norm`; (c) z = [s; W1; dflare_norm(h_k_pre-norm); h_k]. The primary
  reading follows the layer pattern (one `dflare_norm` per consumer of fused context, which gives 4 consumers and
  matches `fuse_weight`'s 4 rows).
- Where `b` is added: after `norm` (as above, which makes it linear and exactly the paper's U+B), or before `norm`
  (`norm(X_out + b)`). [U]
- Cost: U_k for all 8 rows is one GEMM, but each step still needs the full-vocab logits of the previous step to pick
  `prev`. So there are 8 sequential unembed GEMVs (~0.92 GB of bf16 reads each, ~7.4 GB per round). This is the
  dominant drafter cost on the 5090; the 3 layers read only ~0.52 GB. Mitigation, still exact: sample from a
  **restricted q** (for example the top-M of U_k plus the bias computed only on those rows) and pass that same q to
  `accept`. Exactness only needs q to be the distribution the draft was actually drawn from.

## 6. Embedding, softcap, dtype [U]
- Anchor embedding: the HF DFlash generator deliberately **bypasses the target's `embed_norm`**. Its comment says the
  norm "should not be used for DFlash", and Inkling is the model with a NormedEmbedding. That code is generic, so
  this is weak evidence. Primary reading: the **raw** `embed[x0]`. Alternative: x0 = `RMSNorm_w(embed; embed_norm)`.
- Softcap: Inkling applies `30·tanh(x/30)` outside `lm_head` (HF `Inkling2LoopForCausalLM`), so generic HF drafter
  code would skip it. It does not affect greedy drafting (it is monotone). For sampled q it matters. Primary: apply it,
  because the TV loss matched the softcapped target p.
- Precision: bf16 GEMMs; fp32 norms, softmax and RNN gates; bf16 hidden between layers. Same pattern as the target.

## 7. State kept across rounds
- **Per-layer context K/V cache** (post-norm, post-RoPE Kc, and Vc) for every target-processed position. Layers 0 and
  2 need only the last 1024 positions (a ring). **Layer 1 is global and grows with the context**: 2·8·128·2 B =
  4 KiB/token, which is 4 GiB bf16 at 1M (2 GiB in fp8). Cap or disable it at large contexts (DESIGN §3.4). This only
  costs acceptance.
- After each verify, append ctx rows for the **anchor plus the accepted drafts** (their taps come out of the verify
  forward; HF DFlash does exactly this, n_accepted + 1 rows). Rows of rejected drafts are never written, so there is
  **no rollback**. The block's own K/V are transient: computed every round and discarded.
- Head: s resets to 0 every round. `e_ctx` = the latest head ctx row. Nothing else persists. The next anchor is the
  bonus or correction token from verify.
- First round, after prefill: build the ctx caches from the prompt taps. This needs the taps of every prefill chunk
  (6 small 2304→1024 GEMMs per token). Prefix park/restore must also save the drafter ctx state, or rebuild it.
- Lifecycle: prefill → the target emits t0 (anchor, position P) → draft 8 → verify 9 rows [x0, d1..d8] at
  positions P..P+8 → accept n → emit n + 1 → append ctx for positions P..P+n → new anchor at P+n+1.

## 8. Verification (spec.py, unchanged API) [C: paper + HF]
- Feed T = k + 1 rows (k ≤ 8; `Width(kmax=8)` may truncate to a prefix chosen **before** looking at the drafts,
  which is required for losslessness). Greedy: `accept_greedy(lg, d)`. Sampled: `accept(probs(lg, T, top_p, top_k),
  d, q=q[:k])`. Here q_k is the exact distribution d_k was drawn from: the drafter logits after the same
  temperature/top-k/top-p transform.
- The drafter is **meant for sampled drafting with q** at temperature 1.0 (paper eval; TV-matched training). Greedy
  argmax drafts also work. Greedy drafts can also go through `accept` with point-mass q (q=None), which stays exact
  but accepts less.
- json_schema requests: the grammar mask applies to the target p. Draft tokens that the matcher rejects end the
  draft (DESIGN §3.2). Masking q too is optional; it raises acceptance and stays exact if the masked q is the q passed
  to `accept`.

## 9. Riskiest assumptions (ranked)
1. `rnn_proj` input layout: the 4th 2304-wide input, its order, and which `dflare_norm` it uses (§5).
2. `fuse_weight` orientation and semantics: per-consumer rows, pre-norm of the taps, softmax or not (§3).
3. Tap indexing: direct hs index (input of visits 5/23/41, final normed) versus output-of-visit (§3). The HF blog
   supports direct indexing.
4. Where `b` is added (before or after `norm`), and whether the softcap applies (sampled q only).
5. Anchor embedding raw versus embed-normed. Row k drafts p+k+1 (DSpark) versus DFlash's row-0-dropped convention.
6. RoPE variant and position origin (absolute token index); sliding boundary; SiLU.
7. Context K/V for the anchor's own position: does the block also see ctx row p? Assumed no, since the target has not
   processed x0.

## 10. CPU test plan (no GPU window needed; real weights, slow but fine)
Correctness signal = **acceptance against the target**. A wrong drafter only costs speed, because verification is
exact, so the gate is acceptance, not bit-exactness.
1. Loader: map all 54 tensors and assert shapes and dtypes match §2 (cheap; catches typos).
2. Record target taps: run the CPU target (Seq path) teacher-forced on ~6 fixed prompts (chat, prose, code,
   JSON/tool output; ~300 generated tokens each, from tests/oracle.json or greedy runs). Store hs[5], hs[23], hs[41],
   hs[48] and the target logits per position (the expensive step; do it once and save to disk).
3. Offline drafter eval on the stored taps, at every anchor position p: build ctx from positions < p, draft 8, and
   measure (a) greedy: expected accepted length = the longest prefix where d_k == argmax target at p+k (same per-row
   logits); (b) sampled, T=1: per-position acceptance Σ_v min(p_t, q) = 1 − TV, plus its cumulative product. No RNG
   noise.
4. Ablation grid over the [U] choices (each run is cheap once the taps exist): F vs Fᵀ, tap pre-norm on/off, head
   input (a)/(b)/(c), b before/after norm, raw/normed anchor embed, softcap on/off, hs[k] vs hs[k+1]. The right
   combination should stand out sharply. Expect position-1 greedy acceptance well above 50% and mean τ ≥ 3 on
   code/JSON. A wrong wiring typically gives position-1 acceptance below 10%. Published DSpark numbers on other targets
   (τ ~ 4–6 at γ = 7) give a rough bar.
5. Sanity checks on the winner: drafter row 0 alone (no head) should already be decent. Adding the head must raise
   acceptance at positions 2..8, otherwise the head wiring is wrong.
6. End-to-end CPU: spec loop with the drafter on 2 prompts, greedy: output identical to plain greedy (already
   guaranteed by accept_greedy, so this tests the plumbing: positions, ctx append of accepted rows, anchor handoff).
   Sampled: compare the token histogram over many seeds on a short prompt against plain sampling (spec.accept is
   already tested to be exact).
7. Later, in a GPU window: tok/s with and without drafting at ~1k and ~40k context. Decide the layer-1 ctx cap and the
   restricted-q M.

## 11. Measured (2026-11-22, CPU, tests/dspark_ablate.py) — settles most [U] items
Data: 6 teacher-forced prompts from tests/expert_stats.py (code1, code2, chat1, chat2, prose1, json1; 2,555 tokens).
The CPU target recorded the h entering every visit, the output of visit 47 and the final normed state
(~34 min of CPU prefill, `~/llm/logs/ink-dspark-taps.pt`, 516 MB). Target logits = softcap(final @ unembed).
Metric, per draft position k: greedy `a_k` = P(drafter argmax == target argmax | drafts < k matched and the text agreed
with the target there, so a teacher-forced prev equals the real draft). This is a chained estimator of greedy acceptance,
with τ = 1 + Σ Π a. Sampled: `1−TV` = Σ min(p, q) at T = 1, with τ_s from its products. The drafter ran in fp32 for the
eval.

**Verdict, now `dspark.Cfg()` defaults:**
- Taps = **direct hs indices** (h entering visits 5, 23, 41 + final normed) [§3, risk 3: confirmed]. hs[k+1] gives a1
  0.33, (4,22,40) gives 0.42, and visit-47 output instead of the final normed state gives 0.45, against 0.51
  (backbone only).
- Fusion = **DFlare layer-wise fusion** (arXiv 2606.02091, Eq. 3; the DSpark paper cites it, and it names
  `dflare_norm`): `ctx_c = RMSNorm_w(Σ_j softmax(F[c,:])_j · tap_j; dflare_norm_c)`. Raw taps (no pre-norm), **rows**
  = consumers (layer 0, 1, 2, head) [§3, risk 2: confirmed]. Without the softmax a1 is 0.28; with F transposed it is 0.36.
- Context keys **need RoPE** at absolute positions t; without RoPE a1 is 0.05. NeoX rotate-half is correct; GPT-J
  interleaved gives 0.12. A ctx position offset of ±1 gives 0.40 / 0.44.
- Anchor = **embed_norm'd** embedding (as the target's own input); raw gives 0.35. mask_embed is unscaled (×0.5 or ×2 → 0.41).
- **Row convention = DFlash, not the DSpark paper**: row 0 (the anchor) is dropped, and row j (j=1..7) drafts
  position p+j. That gives **7 drafts per round** from block 8. The paper's row-0 convention: a1 0.04 [risk 5: settled].
- Head: z = [s; W1[prev]; h_j; x4] in that order, chunks g, c, o, b = hidden_out(tanh o) added **after** `norm` [§5]
  [risk 1/4: confirmed]. Other z orders give τ ≤ 3.15; gco orders other than g,c,o give τ ≤ 3.63; bias before the norm
  gives τ 2.37. The head lifts τ from 1.71 (backbone only, and its rows ≥ 5 never match) to **3.78**.
- **The 4th 2304-wide rnn input is unresolved and practically inert.** Readings (a) head ctx feature, (b) dflare_norm(E[prev]),
  (c) dflare_norm(pre-norm row) and zeros all give τ 3.78–3.80 and a1 0.662–0.666. That slab's weights have about 1/3 the RMS
  of the h slab. Default = (a).
- Softcap on: tv1 0.675 vs 0.658 off (greedy unaffected).
- Not measurable here (prompts < 1024 tokens): the sliding boundary (< vs <=) and whether a 1024-key window is needed.

**Acceptance of the default config (stride 4, 583 anchors):**
| prompt | a1 | a2..a7 (greedy, chained) | τ greedy (7 drafts) | 1−TV per position 1..7 | τ_s (T=1) |
|---|---|---|---|---|---|
| code1 | 0.88 | .90 .85 .88 .89 .77 .86 | 5.24 | .87 .83 .74 .76 .71 .67 .65 | 4.16 |
| code2 | 0.83 | .88 .84 .81 .74 .81 .76 | 4.53 | .82 .77 .76 .69 .60 .55 .55 | 3.64 |
| json1 | 0.74 | .85 .96 .98 .92 .83 .79 | 4.90 | .76 .67 .64 .63 .58 .55 .47 | 3.03 |
| chat1 | 0.44 | .29 .67 .50 (few) | 1.73 | .42 .28 .25 .27 .29 .25 .20 | 1.58 |
| chat2 | 0.49 | .53 .64 .57 (few) | 2.01 | .53 .46 .45 .31 .24 .27 .32 | 1.93 |
| prose1 | 0.38 | .33 .50 (few) | 1.63 | .46 .34 .24 .26 .27 .24 .19 | 1.66 |
| **all** | **0.66** | .78 .85 .86 .85 .79 .82 | **3.78** | .68 .59 .55 .53 .48 .45 .43 | **2.51** |
Caveats: chat and prose are human-written text fed through the target (teacher-forced), not the target's own output. The
target agrees with that text less often, so acceptance there is pessimistic. The chained a_k for k ≥ 2 only counts anchors
where the text agreed with the target, which may make it somewhat optimistic. The drafter was trained against the official
target; this target is abliterated (small tap shift). Grid log: `~/llm/logs/ink-dspark-grid.log` / `ink-dspark-taps-grid.json`.

**Cost note for §5 (revised):** 7 sequential head steps. Under bias = post, logits_j = U·h_j + (U·W_out)·tanh(o_j) + U·b_out
exactly. So precompute U' = U @ hidden_out.weight ([200058, 256], 102 MB bf16) and c = U @ hidden_out.bias. A round then
reads U once for all 7 rows (0.92 GB) plus 7 × 102 MB, about 1.6 GB, instead of 7 × 0.92 = 6.4 GB. The draft stays exact
(same q). The restricted-q top-M trick is still available on top of that.
