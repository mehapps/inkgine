# Inkling 2 Mini: inference implementation spec

Sources: the official HF repo `thinkingmachines/Inkling-2-Mini` (config, quant config, safetensors headers,
tokenizer, chat template, model card), the Transformers PR #51050 reference (`inkling2_loop`, merged at
commit `430050f6`), and the HF and Thinking Machines launch blogs. "HF ref" means that PR's eager PyTorch path,
the bit-exactness target. Items marked **UNCONFIRMED** are not pinned down by these sources.

Weights we run: `OpenUncensored/Inkling-2-Mini-Uncensored`. Its `model.safetensors.index.json`,
`hf_quant_config.json` and both safetensors headers (`model-00001-of-00001.safetensors`, `mtp.safetensors`)
are **byte-identical** to the official ones: same names, dtypes, shapes and data offsets. Its `config.json`
only lacks the top-level `"dtype": "bfloat16"` key. The model card says only the 20 sliding-layer
`attn.wo_ud.weight` tensors changed (rank-2 abliteration). The loader needs no changes.

## 0. Config at a glance

```
hidden H=2304, layers 24, vocab rows 201024 (embed/unembed), logits 200058 (unpadded)
attn: 24 q heads, 8 kv heads (GQA 3:1, q head h uses kv head h//3), head_dim 128 (q: 3072, kv: 1024)
full-attention layers F = {5,11,17,23}; all others sliding window 1024
attn-res blocks of 6 layers: boundaries at layers {0,6,12,18}
loop: layers 6..17 run 3 passes (48 layer visits per token); loop gate rank 64
MoE: 128 experts, top-8, expert intermediate I=1152, no shared expert, route_scale 8
rms eps 1e-6 everywhere in the text model; no RoPE in the text model (position comes only from rel-bias + log-scaling)
experts: layers 0..19 NVFP4, layers 20..23 MXFP8; everything else bf16 (some f32)
```

## 1. Tokenizer and chat format

`tokenizer.json`: byte-level BPE (o200k-style split regex, `ignore_merges=true`, 199,998 regular
entries), no normalizer, **no BOS is added** (the post-processor is plain ByteLevel). Special tokens:

| id | token | id | token |
|---|---|---|---|
|199999|`<\|endoftext\|>` (pad)|200008|`<\|content_thinking\|>`|
|200000|`<\|message_user\|>`|200010|`<\|end_message\|>`|
|200001|`<\|message_model\|>`|200022|`<\|content_tool_error\|>`|
|200002|`<\|message_system\|>`|200024|`<\|content_xml\|>`|
|200003|`<\|message_tool\|>`|200028|`<\|begin_of_text\|>` (unused by template)|
|200004|`<\|content_text\|>`|200049|`<\|content_invoke_tool_json\|>`|
|200005|`<\|content_image\|>`|200054|`<\|unused_200054\|>` = **image placeholder**|
|200006|`<\|content_model_end_sampling\|>` = **EOS**|200057|`<\|content_invoke_tool_text\|>`|

Other `<|unused_N|>` ids are also special. All special ids are < 200058, so they all have logits.

**Message grammar.** Each content part is its own message: `ROLE [name] CONTENT_TAG body <|end_message|>`.
ROLE is `<|message_user|>`, `<|message_model|>`, `<|message_system|>` or `<|message_tool|>`. A message with
several parts repeats the ROLE token for every part. Rendering, in order:
1. If tools are given: `<|message_system|>tool_declare<|content_xml|>` + JSON array + `<|end_message|>`. Each
   entry is `{"description","name","parameters","type"}` (type defaults to "function"), serialized with sorted
   keys, separators `,` `:`, non-ASCII kept as-is.
2. Messages in order. **Right before the first non-system message** (or at the very end if every message is
   a system message), the effort message is emitted:
   `<|message_system|><|content_text|>Thinking effort level: E<|end_message|>` (see §10).
3. user/system text: `ROLE<|content_text|>text<|end_message|>`.
   Image part (user or tool messages only): `ROLE<|content_image|><|unused_200054|><|end_message|>`. The
   processor then expands that one placeholder into `rows*cols` copies of id 200054 (§8).
4. Tool result: `<|message_tool|>{tool_name}<|content_text|>result<|end_message|>`. The name comes from
   `message.name` or from the matching `tool_call_id`.
5. Assistant (history): optional `<|message_model|><|content_thinking|>{reasoning}<|end_message|>`, then
   `<|message_model|><|content_text|>{content}<|end_message|>`. The text part is skipped when it is empty
   and the message has reasoning or tool calls. Then each tool call is
   `<|message_model|>{fn}<|content_invoke_tool_json|>{"name":"fn","args":{...}}<|end_message|>`, with args
   keys sorted and compact separators. The message ends with `<|content_model_end_sampling|>`.
6. There is **no generation prompt**: the prompt ends after the last message's `<|end_message|>`, and the
   model itself generates `<|message_model|>`.

Example, user = [image, "What is this?"], default effort:
```
<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_image|>
<|unused_200054|> x (rows*cols) <|end_message|><|message_user|><|content_text|>What is this?<|end_message|>
```
(the newlines above are for display only).

**Output parsing.** Generation is a sequence of messages that stops at EOS 200006:
`<|message_model|>[optional name: chars other than '<']<|content_X|>body<|end_message|>` ... `<|content_model_end_sampling|>`.
- `content_thinking` goes to `reasoning_content`; several are concatenated with no separator; no strip.
- `content_text`, `content_tool_error` and `content_invoke_tool_text` go to `content` (concatenated).
- `content_invoke_tool_json`: the body is JSON `{"name":..., "args":{...}}` and becomes a tool call
  `{name, arguments=args}`. The name after `<|message_model|>` repeats the function name.
- Typical output: `<|message_model|><|content_thinking|>...<|end_message|><|message_model|><|content_text|>answer<|end_message|><|content_model_end_sampling|>`.

## 2. Embedding, final norm, head

```
e = embed[ids]                        # bf16 [T,2304], table [201024,2304]
x0 = RMSNorm_w(e; embed_norm.weight)  # computed in fp32, multiplied by the weight, cast to bf16
x0[image positions] = vision features # replaces the *normed* embedding; no extra norm or scale (§8)
...decoder...
f = AttnRes(output_query, Blocks, P_final)   # fp32, §4
f = RMSNorm_w(f; norm.weight) -> bf16
logits = float32(f @ unembed[:200058].T)     # bf16 GEMM, fp32 output
logits = 30 * tanh(logits / 30)
```
`RMSNorm_w(x; w)`: `xf = fp32(x)`, `y = xf * rsqrt(mean(xf^2) + 1e-6)`, `out = (w * y)` in fp32, then cast to the
activation dtype (bf16). The weight is a plain gain (not `1 + w`). `RMSNorm_nw` is the same with no weight.
Embedding and unembedding are **separate tensors** (not tied). The unembed is stored padded to 201024 rows;
use only the first 200058 (rows ≥ 200058 are padding). Feed the embedding table any id < 201024.

## 3. Decoder layer

Notation: `h` is the bf16 layer input (the partial sum of the current attn-res block), and `Blocks` is the list of
completed block states (bf16 [T,2304] each). `L` is the layer index. `bnd = (L % 6 == 0)`. Sliding layers are
`L ∉ F`; full layers are `L ∈ F`.

### 3.1 Layer skeleton (all ops in order)
```
# attention input (§4). fp32 outputs
if L == 0:      a = h                                   # sliding, no attn-res query
elif sliding:   a = AttnRes(attn_q[L], Blocks, h)       # 1 query
else:           aq, ak, av = AttnRes(attn_q[L] (3 queries), Blocks, h)

# norms → bf16 projection inputs
sliding, no messages: xq = xk = xv = RMSNorm_w(a; attn_norm)
sliding, messages:    xq = RMSNorm_w(MG_q(a, m_q); attn_norm); xk = RMSNorm_w(MG_k(a, m_k); attn_norm); xv likewise (same attn_norm weight)
full:                 xq = RMSNorm_w(aq'; q_input_norm); xk = RMSNorm_w(ak'; k_input_norm); xv = RMSNorm_w(av'; v_input_norm)
                      where aq' = MG_q(aq, m_q) if messages else aq (likewise for k, v)

o = Attention(xq, xk, xv)                 # bf16 [T,2304], §3.2
o = ShortConv(o; attn_sconv)              # §3.3, bf16 out

if bnd: Blocks' = Blocks + [h];  r = fp32(o)
else:   Blocks' = Blocks;        r = fp32(h) + fp32(o)
m = AttnRes(mlp_q[L], Blocks', r)         # r is passed in fp32 (NOT rounded to bf16)
if messages: m = MG_mlp(m, m_mlp)
y = MoE(RMSNorm_w(m; mlp_norm) -> bf16)   # bf16, §5
y = ShortConv(y; mlp_sconv)               # bf16
return bf16(r + fp32(y))                  # the new h (bf16); the caller also does Blocks = Blocks'
```
The hidden state between layers is bf16. Messages exist only for loop layers in passes 2 and 3 (§6).

### 3.2 Attention (same structure for sliding and full; differences flagged)
Inputs: `xq` (it also drives the rel-bias, sink and output gate), `xk`, `xv`; all bf16 [T,2304]. Positions
`pos` are absolute token indices (0-based; image tokens count like any other token).
```
Q = xq @ wq_du.T  -> [T,24,128]                    (bf16 GEMM)
Q = fp32(Q) * q_l2_clip.scale[24,128]               # per-head, per-dim gain
Q = Q * min(1, 22.627417 / (||Q||_2 + 1e-6))        # per head vector; 22.627417 = 16*sqrt(2)
Q = bf16(Q)
FULL only: Q = bf16(fp32(Q) * tau(pos)),  tau(p) = 1 + 0.1 * ln(max((p+1)/8192, 1))

K = xk @ wk_dv.T  -> [T,1024]; K = KConv(K, xk); view [T,8,128]
K = bf16( fp32(K) * min(1, 22.627417/(||K||+1e-6)) )     # no gain, no tau
V = xv @ wv_dv.T  -> [T,1024]; V = VConv(V, xv); view [T,8,128]
V = bf16( RMSNorm_nw(fp32 V) per head over 128 dims, eps 1e-6 )   # "value norm, no gamma"
# KConv/VConv: sliding = static ShortConv (k_sconv.weight / v_sconv.weight); full = DynamicConv (§3.3).
# Conv runs on the 1024-wide linear output, BEFORE the head split / clip / norm. Cache K,V after clip/norm.

# relative position bias (per query token, per head)
R = xq @ wr_du.T -> [T,24,16];  R = fp32(R) * rel_l2_clip.scale[24,16]
R = R * min(1, 8.0/(||R||+1e-6));  R = bf16(R)
FULL only: R = bf16(fp32(R) * tau(pos))
table = R @ rel_logits_proj.proj [16,1024]  -> [T,24,1024]     # bf16
bias[h,i,j] = table[i,h,d] if 0 <= d < 1024 else 0,  d = pos_i - pos_j   # rel_extent 1024 in both layer types

sink[h,i] = (xq @ attn_sink_proj.T)[i,h]                 # [T,24], one learned logit per token and head

S[h,i,j] = (Q_i · K_j) * 128^-0.5 + bias[h,i,j]          # HF ref: bf16 matmul, bf16 add
mask: causal j <= i; sliding layers also need  j > i - 1024  (1024 keys including self)
P = softmax_fp32([S[h,i,:], sink[h,i]])[:-1]             # the sink joins the softmax denominator only (zero value)
A = bf16(P) @ V  -> [T,24,128] -> [T,3072]
g = sigmoid(fp32(xq @ gate_attn.T))                       # [T,3072]
o = bf16(fp32(A) * g) @ wo_ud.T                           # [T,2304], no biases anywhere
```
The sink is NOT scaled by 128^-0.5. The softmax runs in fp32 and the result is cast to bf16 before multiplying with V.
With the clips, |Q·K| ≤ 512, so pre-bias logits stay within ±45.3 (up to ±50 with tau). There is no softcap on attention logits.

### 3.3 Short convolutions (kernel 4, depthwise, causal, with residual and fixed rescale)
All math is fp32; the output is cast back to the input dtype (bf16). There are four instances per layer visit:
`k` (1024 ch), `v` (1024 ch), `attn_sconv` (2304 ch, on the o_proj output) and `mlp_sconv` (2304 ch, on the MoE output).

**Static** (`*.weight`, shape [C,1,4]; all attn_sconv/mlp_sconv, plus k/v in sliding layers):
`out[t,c] = (Σ_{i=0..3} w[c,i] * x[t-3+i, c]  +  x[t,c]) / sqrt(5)`.
`w[c,3]` multiplies the current token. Missing past (t-k < 0, start of sequence) counts as 0.

**Dynamic** (k/v in full layers 5,11,17,23): `static_w` [4,1024], `weight_proj` [32,2304]:
```
dyn = fp32(x_in @ weight_proj.T) -> [T, 8 heads, 4 taps]     # x_in = xk (for K) or xv (for V), bf16 GEMM
W[t, c, j] = static_w[j, c] + dyn[t, c // 128, j]            # j = backward lag; j=0 is the current token
out[t,c] = (Σ_{j=0..3} W[t,c,j] * x[t-j, c]  +  x[t,c]) / sqrt(9)   # = /3
```
**Conv cache state** per instance and per layer visit: the last 3 conv *inputs* (pre-conv values, e.g. the raw
`xk@wk_dv.T` rows). These are bf16-exact, so storing them in bf16 is lossless. Decode: out_t uses state[0..2] + x_t, then shift.

## 4. Attention residuals ("attn_res", Kimi-style)

The residual stream is not one running sum. Each 6-layer block accumulates its own partial sum `h` (reset at block
boundaries), and finished blocks are kept as a list. Every layer input (attention and MLP side) is a learned softmax
mix over [finished blocks..., current partial]:
```
AttnRes(q[nq,2304] fp32, Blocks (list of n), p):
    S = stack(Blocks + [p]) as fp32  -> [n+1, 2304]       (per token)
    Sn = RMSNorm_nw(S) per entry (eps 1e-6)                # keys AND values are normalized (attn_res_normalize_values=true)
    w = softmax_over_entries( (q · Sn_k) / sqrt(2304) )    # fp32
    return Σ_k w_k Sn_k                                    # fp32; one output per query row
```
The same query applies at every token position; there is no causal mixing across tokens (the mix is over depth only).
Block bookkeeping (the boundary layer L appends its *input* `h` to Blocks):
- L=0: Blocks=[] → layer 0 does Blocks=[x0] (the embedding), and its partial restarts from attn out.
- Before layer 6: Blocks=[x0]; layer 6 appends B1 = its input. Layer 12 appends B2. Layer 18 appends B3.
- Final: `AttnRes(output_query, [x0,B1,B2,B3], P)` where P = output of layer 23, then norm and head (§2).

Query tensors (fp32): `attn_res.attn_queries.{i}` belongs to **layer i+1** ([2304], or [3,2304] = (q,k,v) rows for
full layers, i.e. i=4,10,16,22). `attn_res.mlp_queries.{i}` belongs to layer i. `attn_res.output_query` is the final mix.

## 5. MoE

```
u = RMSNorm_w(m; mlp_norm) -> bf16 ; uf = fp32(u)
logits = uf @ fp32(gate.weight).T            # [T,128], fp32 GEMM (weight stored bf16, upcast)
s = sigmoid(logits)
idx = top8(s + gate.bias)                    # gate.bias f32 [128]: selection only
w = s[idx] / (Σ s[idx] + 1e-20) * 8.0        # weights use the UNbiased scores
out = Σ_k w_k * Expert_{idx_k}(u)            # accumulated in fp32, then cast to bf16
Expert(u): gu = u @ W13.T  [2304 = gate 1152 ‖ up 1152 after de-interleave, §7]   (bf16 GEMM, bf16 out)
           g, v = fp32(gate), fp32(up)
           act = (4*tanh(g/4)) * sigmoid(g) * (25*tanh(v/25))    # sigmoid sees the raw g; caps bound the linear factors
           y = bf16(act) @ W2.T   [W2: 2304 x 1152]
```
The HF ref uses bf16 activations with dequantized weights. The training/runtime recipe declares NVFP4 per-token
activation quantization for NVFP4 experts and MXFP8 activation quantization for MXFP8 experts
(`required_activation_quantization` in hf_quant_config). Matching the HF ref means **not** quantizing activations.
Whether quantized activations match the official Tinker serving more closely is UNCONFIRMED.

## 6. Loop (layers 6..17, 3 passes)

```
h = x0; Blocks = []
for L in 0..5:  h = layer_L(h, Blocks)                # Blocks becomes [x0]
h_pre, Blocks_pre = h, Blocks                          # Blocks_pre = [x0]
pass 1: for L in 6..17: h = layer_L(h, Blocks)        # no messages; Blocks -> [x0,B1,B2]
for p in 2..3:
    Bprev = Blocks[2]          # B2 of the previous pass (sum of layers 6..11 of that pass = input of its layer 12)
    Pprev = h                  # output of layer 17 of the previous pass (partial of layers 12..17)
    h, Blocks = h_pre, Blocks_pre                      # every pass restarts from the pre-loop state
    for i, L in enumerate(6..17):
        (m_q, m_k, m_v, m_mlp) = AttnRes(MQ[i] (4 x 2304), [Bprev], Pprev)   # same AttnRes math, 2 entries
        h = layer_L(h, Blocks, messages)
# after pass 3: Blocks = [x0,B1,B2(pass3)], h = pass-3 output
for L in 18..23: h = layer_L(h, Blocks)
```
Messages depend only on the previous pass, so all 12×4 can be computed at the start of the pass.
`MQ` is built from `middle_cycle_cross_pass_queries` [24,2304] (rows 2i = q-message query, 2i+1 = mlp-message query
of loop layer i) and `middle_cycle_cross_pass_kv_queries` [24,2304] (rows 2i = k, 2i+1 = v). So
`MQ[i] = [cpq[2i], cpkv[2i], cpkv[2i+1], cpq[2i+1]]` = (q, k, v, mlp).

**Loop gate** `MG(x, msg)` (controllers.{attn→q, k, v, mlp}, per loop layer):
```
x, msg in fp32
z = concat(RMSNorm_nw(x), RMSNorm_nw(msg))            # [4608], order: current input first, then message
z = fp32( bf16(z) @ down_weight.T )                    # down_weight [64,4608] bf16 GEMM
f = tanh(z @ fp32(forget_weight).T)                    # [2304], forget_weight [2304,64], fp32 math
s = tanh(z @ fp32(gain_weight).T)                      # gain_weight [2304,64], fp32 math
return (1 - f) * x + s * msg                           # fp32
```
Placement: sliding loop layers gate the single attn-res output `a` three times (q/k/v) before the shared
`attn_norm`. Full loop layers (11, 17) gate `aq/ak/av` separately before their own q/k/v input norms. `MG_mlp`
is applied to the mlp-side AttnRes output before `mlp_norm`.

**State per pass:** every layer visit has its own KV cache and its own 4 conv states. That is 48 attention caches
(layers 0..5, then 6..17 ×3, then 18..23). Passes share weights only. Each loop pass sees different K/V inputs,
so its caches really differ.

## 7. Weight storage (`model-00001-of-00001.safetensors`, prefix `model.llm.`)

| tensor (per layer `layers.N.`) | dtype | shape | notes |
|---|---|---|---|
| `embed.weight`, `unembed.weight` | BF16 | [201024,2304] | unembed: use rows < 200058 |
| `embed_norm.weight`, `norm.weight` | BF16 | [2304] | |
| `attn_norm.weight` | BF16 | [2304] | sliding layers only (20) |
| `{q,k,v}_input_norm.weight` | BF16 | [2304] | full layers 5,11,17,23 |
| `mlp_norm.weight` | BF16 | [2304] | all |
| `attn.wq_du.weight` / `attn.gate_attn.weight` | BF16 | [3072,2304] | Q / output gate |
| `attn.wk_dv.weight`, `attn.wv_dv.weight` | BF16 | [1024,2304] | |
| `attn.wo_ud.weight` | BF16 | [2304,3072] | (20 sliding ones differ in the uncensored model) |
| `attn.wr_du.weight` | BF16 | [384,2304] | rel states, 24×16 |
| `attn.rel_logits_proj.proj` | BF16 | [16,1024] | |
| `attn.q_l2_clip.scale` / `attn.rel_l2_clip.scale` | BF16 | [24,128] / [24,16] | |
| `attn.attn_sink_proj.weight` | BF16 | [24,2304] | |
| `attn.{k,v}_sconv.weight` | BF16 | [1024,1,4] | sliding layers |
| `attn.{k,v}_sconv.static_w` / `.weight_proj` | BF16 | [4,1024] / [32,2304] | full layers; weight_proj rows = head*4+tap |
| `attn_sconv.weight`, `mlp_sconv.weight` | BF16 | [2304,1,4] | post-attn / post-MoE conv |
| `mlp.gate.weight` / `mlp.gate.bias` | BF16 / F32 | [128,2304] / [128] | router (compute in fp32) |
| `controllers.{attn,k,v,mlp}.down_weight` | BF16 | [64,4608] | layers 6..17 |
| `controllers.*.forget_weight`, `.gain_weight` | BF16 | [2304,64] | upcast to fp32 |
| `attn_res.attn_queries.{0..22}`, `mlp_queries.{0..23}`, `output_query` | F32 | see §4 | |
| `middle_cycle_cross_pass_{,kv_}queries` | F32 | [24,2304] | §6 |
| `mlp.experts.w13_weight` (NVFP4, L0..19) | U8 | [128,2304,1152] | 2 fp4 codes per byte |
| `mlp.experts.w13_weight_scale` (NVFP4) | F8_E4M3 | [128,2304,144] | 1 scale / 16 elements of a row |
| `mlp.experts.w2_weight` / `_scale` (NVFP4) | U8 / F8_E4M3 | [128,2304,576] / [128,2304,72] | W2 is [2304 out,1152 in] |
| `mlp.experts.w{13,2}_weight_amax` | F32 | [128] | per expert (NVFP4 only) |
| `mlp.experts.w{13,2}_weight_outer_scale_denominator` | F32 | [1] | per tensor (NVFP4 only) |
| `mlp.experts.w13_weight` / `_scale` (MXFP8, L20..23) | F8_E4M3 / U8 | [128,2304,2304] / [128,2304,72] | 1 E8M0 scale / 32 elements |
| `mlp.experts.w2_weight` / `_scale` (MXFP8) | F8_E4M3 / U8 | [128,2304,1152] / [128,2304,36] | |

Vision tensors are under `model.visual.` (§8). The MTP drafter is in a separate `mtp.safetensors`, not in the index (§9).

**NVFP4 dequant** (exact, as in the HF ref: fp32 math, then one cast to bf16):
`E2M1 = [0,.5,1,1.5,2,3,4,6, -0,-.5,-1,-1.5,-2,-3,-4,-6]` (code bit 3 is the sign).
Byte `b` at column k of a row holds element `2k` in the **low nibble** (`b & 0xF`) and element `2k+1` in the high
nibble (`b >> 4`). `W[e,r,c] = E2M1[code] * fp32(scale_e4m3[e, r, c//16]) * (amax[e] / denom[0])`.
**MXFP8:** `W[e,r,c] = fp32(w_e4m3[e,r,c]) * 2^(scale_u8[e,r,c//32] - 127)`. hf_quant_config lists
weight_block_shape [32,32], but the stored scales are per row per 32 columns (shape above). Follow the shapes.
**w13 row order is interleaved:** stored row `2i` = gate row i and row `2i+1` = up row i (i < 1152). De-interleave
(or index rows accordingly) to get gate = W13[0::2], up = W13[1::2].
The quant config's `scale_selection_rule: fouroversix_mae` describes how the weights were quantized. It does not affect dequant.

## 8. Vision

**Preprocessing** (`Inkling2ImageProcessor`; there is **no resize** and no min/max-pixel rule in the HF ref):
1. Convert to RGB. RGBA is composited (integer math) onto a checkerboard of 8×8-px cells, values 204 where
   ((y//8)+(x//8)) is even and 153 otherwise: `rgb = (c*α + bg*(255-α) + 127) // 255`.
2. `rows = ceil(H/40)`, `cols = ceil(W/40)`. Pad bottom/right to `40*rows × 40*cols` with the value **-1** (on the 0–255
   scale), i.e. -1.00784 after normalization.
3. `x = (v/255 - 0.5)/0.5` → [-1,1] (padding → -1.00784).
4. Cut into `rows*cols` tiles of 40×40×3, in raster order over tiles. Each tile is one LM image token. Layout is
   `[tile, 40(y), 40(x), 3]`. `image_grid_thw = (1, rows, cols)`.
Image token count = `rows*cols` (e.g. 1024×768 → 26×20 = 520). The model card recommends sides of 40–4096 px. A cap is
UNCONFIRMED (you may want one for memory).

**ViT** (bf16; hidden 1024, 23 layers, 16 heads × 64, MLP 4096, exact erf-GELU, LayerNorm eps 1e-5):
```
each tile -> 2x2 native patches of 20x20 (order: (0,0),(0,1),(1,0),(1,1) within the tile)
patch vector = flatten(20 y, 20 x, 3 c) -> 1200;  emb = patch @ patch_embedding.weight[1024,1200].T  (no bias)
patch grid per image: hp = 2*rows, wp = 2*cols; patch (r,c) in native-patch units
pos-emb: table position_embedding [1024 = 32x32, 1024]; bilinearly resample the 32x32 grid to hp x wp,
         align_corners=False: src = (idx+0.5)*32/size - 0.5, taps floor and floor+1 clamped to [0,31],
         weights 1-|src-tap| (clamped >= 0), separable in r and c (4 taps); add to emb
sequence per image: [CLS, patches in tile-major order (tile raster, then 2x2 inside)]
CLS = class_embedding + cls_position_embedding
h = LayerNorm(seq; ln_pre)
2D RoPE: theta 10000, per head dim 64. Positions: patch (r+1, c+1), CLS (0,0).
  inv = 10000^(-(0,2,..,30)/32)  (16 freqs); angles_w = c*inv, angles_h = r*inv
  cos/sin vector (64) = repeat_each_twice(concat(angles_w, angles_h)); rotate interleaved pairs (x0,x1)->(-x1,x0)
  q' = q*cos + rot(q)*sin (fp32), same for k
23 x [ h = h + ls_1.gamma * Attn(LN1(h));  h = h + ls_2.gamma * MLP(LN2(h)) ]
   Attn: q/k/v/out_proj with biases; full bidirectional attention within one image only; scale 64^-0.5; softmax fp32
   MLP: c_fc (1024->4096, bias) -> GELU(erf) -> c_proj (4096->1024, bias)
(no final LayerNorm)
drop CLS; concat the 4 patch states of each tile (in 2x2 order) -> [tiles, 4096]
proj: y = GELU( x @ proj_in.weight[4096,4096].T ) @ proj_out.weight[2304,4096].T    (no biases)
```
The `y` rows replace the embeddings at the 200054 positions, in order (images in prompt order, tiles in raster order).
`temporal_patch_size=2` is unused for images (the HF ref has no temporal dim; patch_embedding in=1200). Video
handling is UNCONFIRMED and not needed.
Vision tensor names: `model.visual.backbone.embeddings.{patch_embedding.weight, position_embedding, class_embedding,
cls_position_embedding}`, `backbone.ln_pre.{weight,bias}`, `backbone.layers.N.{ln_1,ln_2}.{weight,bias}`,
`.ls_{1,2}.gamma`, `.self_attn.{q,k,v,out}_proj.{weight,bias}`, `.mlp.{c_fc,c_proj}.{weight,bias}`,
`model.visual.proj_in.weight`, `model.visual.proj_out.weight` (all BF16).

## 9. DSpark / MTP drafter (high level; HF ref does not implement it)

`mtp.safetensors` (bf16, ~0.3B params) holds a 3-layer dense drafter (`mtp_config`): hidden 2304, 24/8 heads × 128,
dense SwiGLU 9216, RoPE theta 1e6 (unlike the target), layers 0 and 2 sliding 1024. It has per-head q/k RMSNorm and
separate context K/V projections (`k_proj_ctx`, `v_proj_ctx`, `k_norm_ctx`) that read target hidden states.
`fuse_weight` [4,4] mixes 4 target features: the inputs to decoder visits 5, 23 and 41 and the final normalized state
(visit index 48), per `target_layer_ids [5,23,41,48]`. `mask_embed` fills the not-yet-known positions of a block of 8
(block_size 8, drafts several tokens in parallel). A `markov_head` (rank 256, `markov_w1` [201024,256] token
embedding, `rnn_proj` [768,5120], `hidden_out`) adds an RNN-style correction across the drafted tokens. Exact math is
UNCONFIRMED. This is optional speculative decoding, so skip it for a first engine.

## 10. Generation defaults and reasoning effort

`generation_config.json`: `do_sample=true, temperature=1.0`, no top-p/top-k given, `eos=200006`, `pad=199999`. Stop
on 200006. Max context 1,048,576.
Reasoning effort is **text in the prompt**, not a sampling parameter: the system message `Thinking effort level: E`
(placement in §1). Named levels map as none→`0`, minimal→`0.1`, low→`0.2`, medium→`0.7`, high→`0.9` (**default**
when unspecified), xhigh→`0.99`, max→`0.99`. Any number in [0, 0.99] is allowed and printed with Jinja float
formatting (e.g. `0.35`); exactly 0 prints as `0`. Whether effort 0 produces no `content_thinking` at all is
UNCONFIRMED (the uncensored card implies "none" gives short answers).

## 11. Most likely sources of subtly wrong output

1. **Attn-res block bookkeeping**: the boundary layer's *input* is appended to Blocks, and at a boundary the residual
   restarts at the attention output (h is NOT added). The MLP-side mix already includes that new block. The final
   output goes through `output_query` AttnRes before `norm`.
2. **Off-by-one attn query index**: `attn_queries.i` → layer i+1, but `mlp_queries.i` → layer i.
3. **Normalized values in AttnRes**: the mix returns a weighted sum of the *RMS-normalized* states, not the raw ones.
   Scale is 1/sqrt(2304).
4. **Loop restart**: passes 2 and 3 start from the pre-loop `h` and `[x0]`, not from the pass-1 output. The only
   cross-pass channel is the messages from [B2_prev, layer-17 output_prev]. Separate KV/conv caches per pass.
5. **Message query order**: (q, k, v, mlp) = (cpq[2i], cpkv[2i], cpkv[2i+1], cpq[2i+1]). Gate output is
   `(1-tanh f)·x + tanh s·msg`; the concat order is [x, msg].
6. **w13 interleave**: gate/up rows alternate in storage. Also the nibble order (low = even element) and the
   per-expert `amax/denominator` factor.
7. **Sink logit** is per token (`attn_sink_proj(xq)`), unscaled, in the fp32 softmax, with zero value.
8. **Rel-bias / gate / sink use the query-side input `xq`**, even in full layers and loop passes where xk and xv differ.
9. **Conv residual and rescale**: `(conv + x)/sqrt(5)` static and `/3` dynamic. Static tap 3 = current token, dynamic
   tap 0 = current token. The conv applies to K/V before head split, clip and value norm. The dynamic taps come from
   xk/xv, not from the K/V themselves.
10. **Log scaling** applies only in full layers, to Q and the rel states (not K), only at positions >= 8192, and
    after the L2 clip.
11. **Sliding window = 1024 keys including self** (`j > i-1024`). The full layers' rel-bias is zero for distance ≥ 1024.
12. **Router**: sigmoid scores; the bias is used only for selection; renormalize over the top-8; ×8. Run in fp32 with an fp32 weight.
13. **Expert activation**: `softcap4(g)·sigmoid(g)·softcap25(u)`. Not plain SiLU.
14. **Unpadded vocab**: the head has 200058 rows, then the 30·tanh(x/30) softcap. The embedding is RMS-normed (with
    gain) before use, and image features replace the normed embedding.
15. **Precision**: hidden states are rounded to bf16 at every layer output, but `r` inside a layer stays fp32. AttnRes,
    MG, router, convs and softmax run in fp32.
16. **Prompt**: no BOS; effort message before the first non-system message; one role token per content part; image
    placeholder expanded to rows×cols tokens.
