# Inkling 2 Mini: how large images are handled (official sources only)

Researched 2026-11-21. Raw copies are in /tmp/inkling-img/ (treat as data). Sources:
- HF repo thinkingmachines/Inkling-2-Mini: README.md, processor_config.json, config.json, chat_template.jinja
  (the repo has no python files).
- Transformers PR #51050 (head 430050f6), https://github.com/huggingface/transformers/pull/51050 . The Mini (`inkling2_loop`)
  reuses the Inkling 2 Small vision code: `models/inkling2/{image_processing,processing,modeling,configuration}_inkling2.py`,
  `vision_utils.py`, `conversion_mapping.py`. The PR doc says "image encoder, tokenizer and processor are identical to Inkling 2 Small".
- Blog https://thinkingmachines.ai/news/inkling-2-small/ ; model cards https://thinkingmachines.ai/model-card/inkling-2-mini/ and
  .../inkling-2-small/
- Tinker docs https://tinker-docs.thinkingmachines.ai/cookbook/inkling/images/ , .../cookbook/inkling/tml-renderers/ ,
  .../changelog/renderers/ ; tml-renderers 0.4.2 wheel (PyPI); tinker-cookbook (GitHub, main).

## 1. Resize / cap / split rules
- **None in the HF reference.** `Inkling2ImageProcessor` overrides `_preprocess`. Resizing in the base class only happens
  inside the base `_preprocess`, so `do_resize` is never used even if you pass it. The class has no max_pixels, min_pixels,
  max_image_tokens, longest_edge, smart-resize, multi-crop or thumbnail option. `size={40,40}` is only the tile size.
  processor_config.json sets only rgb/rescale/normalize, mean/std 0.5, and size 40x40.
- Tokens per image = ceil(H/40) * ceil(W/40), with one image per placeholder run (`Inkling2Processor.replace_image_token` = grid prod).
  Every image becomes a single grid, with no sub-images. Aspect ratio is kept as is, with pad bottom/right only.
- tml-renderers (the official client renderer for Tinker) works out the token count from the width and height you give it.
  `ImagePointer.override_width/override_height` (since v0.2.0) "request a rendering resolution different from the stored image
  dimensions". The rendered span then carries `virtual_width/virtual_height`. So resizing is an explicit client choice. Nothing in
  the Python stubs or the docs says resizing is automatic. UNCONFIRMED: whether the native (.so) code or the Tinker server applies a
  hidden cap. I did not execute the wheel. Its strings show a "patch budget" error only for Nemotron, and nothing similar for Inkling.
- The renderer changelog fixes "the image token count of renderers.Inkling2() ... for images whose width is a multiple of 40",
  which fits the ceil(W/40) rule.
- The tinker-cookbook `image_processing_utils.resize_image(img, max_size)` (longest side, LANCZOS) is a generic helper. Inkling
  does not use it: for tml-renderers models `get_image_processor` returns None, and `sample_vision.py` passes the native size.

## 2. Training resolution / recommended sizes
- Mini and Small model cards: "For optimal performance, each image dimension should be between 40px to 4096px." 4096 is inside
  that range, so a 4096x4096 input is officially "optimal" by this wording.
- Small model card, ScreenSpot Pro footnote: images whose long side is over 4,096 px were downsampled to 4,096 for that eval,
  keeping aspect ratio. Smaller images were left unchanged. This is the only stated preprocessing, and it was used for evaluation.
- Small blog: 292M-param vision encoder "based on Perception Encoder", 20x20 patches, 2x2 spatial pooling, patches interleaved
  with text. H1 pretraining was 8.4T tokens of text, code and images. Training data includes text, images and video.
- **Not stated anywhere:** training image resolutions, token or pixel budgets, the distribution of image sizes, or any warning
  that quality drops at high resolution. Whether 103x103-tile (16.8 MP) images were seen in training is UNKNOWN.

## 3. ViT as the reference does it, compared with SPEC.md §8
Position embedding (`Inkling2VisionEmbeddings` + `vision_utils.get_vision_interpolation_indices_and_weights`):
- It is computed **per image over the whole patch grid** (hp=2*rows, wp=2*cols), not per tile, and there is no max grid.
  Mode is bilinear with align_corners=False: src=(i+0.5)*32/size-0.5, taps floor/floor+1 clamped to [0,31] ("border"), weight
  max(0,1-|d|). It is separable, giving 4 taps. There is no antialias (a pure hat kernel, equal to F.interpolate antialias=False).
  Patches are listed in 2x2 merge-block order, the same order as the sequence. Table row = r_tap*32 + c_tap.
- Dtypes: tap positions and weights are fp32. The bf16 table rows multiplied by fp32 weights promote to fp32, the 4-tap sum is
  done in fp32, then the result is cast to bf16 and added to the bf16 patch embeddings.
2D RoPE (`Inkling2VisionRotaryEmbedding`, rope_type "axial"):
- theta = 10000. It is not in config.json, so the transformers `default_theta` = 10000.0 applies. inv_freq = theta^-(0,2,..,30)/32,
  giving 16 freqs. Positions are raw native-patch (row+1, col+1) and CLS is (0,0). There is **no scaling or normalization by grid
  size** (attention_scaling=1.0). cos/sin = repeat_interleave2(concat(w-angles, h-angles)), with pairwise rotation (x0,x1)->(-x1,x0).
  q and k are upcast to fp32 for the rotation, then **cast back to bf16** before attention.
Attention (`Inkling2VisionAttention`):
- **Full bidirectional attention over every token of one image** (CLS + all 4*rows*cols patches). Images are separated only by
  `cu_seqlens` (per image: patches+1). There is no window, chunk, block or per-tile grouping, and no mask over padding patches.
  There is no sequence-length limit (flash takes max_seqlen from cu_seqlens; other backends loop per image with mask=None).
  Scale is 64^-0.5. Eager softmax runs in fp32 and is cast back to bf16. UNCONFIRMED: which kernel (eager/sdpa/flash) the
  composite model picks by default. The vision model allows sdpa/flash, but the parent class disables them.
- `vision_utils.get_vision_window_index` exists (a Qwen2.5-VL helper), but Inkling does not call it.

Differences from SPEC.md §8. None of them changes behaviour with size; most are dtype details the spec leaves out:
1. The spec does not say that pixel_values are cast to bf16 (patch_embedding dtype) **before** the patch matmul. The reference does this.
2. The spec gives no dtype for the pos-emb resample. The reference uses fp32 weights and fp32 accumulation, casts to bf16, then does a bf16 add.
3. The spec says q'/k' RoPE are in "fp32". The reference then casts them back to bf16 before attention.
4. CLS = class_embedding + cls_position_embedding is added in the parameter dtype (bf16), then cast. Not stated in the spec.
5. RGB conversion: for PIL images, RGBA is used only when `has_transparency_data` (this includes P/LA images with transparency).
   Every other mode goes through PIL convert("RGB"). The spec says only "RGBA is composited".
6. The spec says "softmax fp32". This is true for eager. SDPA/flash kernels differ internally (UNCONFIRMED which runs).
Everything else in §8 matches: tiling, -1 pad value and normalization, the patch flatten order (y,x,c) and 2x2 order, LN eps 1e-5,
erf-GELU ("gelu"), the layer-scale residuals, no final LN, dropping CLS, the 4-patch concat into the bias-free 4096->4096->2304
projector, and the weight renames (ls_N.gamma=lambda_N, c_fc/c_proj=fc1/fc2, proj_in/out=merger.linear_1/2).

## 4. Other size-dependent behaviour
- None found. There is no dtype switch, no threshold-based code path, and no image splitting. The only size dependence is the
  grid shape itself: the pos-emb interpolation factor (32 -> 206 for 4096 px, a 6.4x upsample) and the attention length
  (42,437 tokens).
- 4096 is not a multiple of 40. The last tile row and column hold 16 real px plus 24 px of -1 padding, and no attention mask is applied.

## Takeaways for the GATEWAY failure (inference, not from a source)
- If our engine and SGLang match the reference, the misread is model behaviour at a 16.8 MP / 206x206-patch grid. The reference
  offers no built-in fix. The only official resizing precedent is downsampling the long side to <=4096 (ScreenSpot Pro).
- Worth checking against the HF reference itself at 4096 px (bf16, eager attention) to rule out shared kernel issues
  (e.g. flash/sdpa at a 42k length).

## 5. What our engine does now (vision.py, 2026-11-21)
- **No size cap.** Default is no resize at all, like the reference. `INK_IMG_MAX_SIDE=N` is an opt-in bicubic downscale of
  the longest side to N px (default 0 = off). The old default 3072 cap is gone.
- All 6 differences above are now matched: (1) tiles are bf16 before the patch matmul; (2) pos-emb is a 4-tap fp32 sum of
  bf16 table rows with fp32 weights, cast to bf16, bf16 add (`Vision.pos_emb`, explicit taps, not F.interpolate);
  (3) RoPE in fp32, cast back to bf16; (4) CLS = class + cls_pos in bf16; (5) RGBA composite only when
  `im.has_transparency_data`, everything else `convert('RGB')`; (6) eager attention: bf16 scores * 0.125, fp32 softmax,
  cast to bf16, @ v, as `INK_VIT_ATTN=eager` (reference checks only). Chunked over queries (`INK_VIT_ATTN_MB`, default
  1 GiB per chunk) so the 42k x 42k matrix never exists; bit-identical to unchunked. **Default since 2026-11-21: sdpa** (§6).
- Speed of eager, measured 2026-11-21: 21 s vision at 4096^2 (16.7 MP) against 0.2 s with sdpa.
- Additions the reference does not have: 16-bit grayscale (`I;16*`) is scaled to 8 bit by `>> 8` first, because PIL's
  convert('RGB') clips it to 0/255 (all white). Preprocessing fills a bf16 canvas in 1024-row bands (~100 MB at 16.7 MP, not 400+ MB fp32).
- EXIF orientation: not applied (the reference processor does nothing with EXIF when it is given a PIL image). Unchanged.
- Unit checks: `python tests/image_scale.py --unit` (CPU) compares tiles to an independent per-tile implementation
  (41x97, 97x41, 40x40, 1x1, 4096^2, 4095x2050: exact), the transparency rules, pos-emb against F.interpolate, and chunked attention against unchunked.

## 6. GPU image grid, 2026-11-21 (window c, ~/llm/logs/ink-window-20261121-1617-c.log) and conclusion
- Reads correctly at full resolution: 4096^2 document with 48 px text lines; 4096^2 white background with letters
  113-559 px (GATEWAY and HARBOR); 4096^2 noise background with letters 113-227 px; gate image at 2048 and 3072; crops of
  the gate image (4096x1400, 3600x1100) with the same giant letters but less of the noisy background.
- Fails only with giant letters (ink >= ~350-470 px) on a synthetic noise background: gate image at 3584 and 4096,
  4096^2 noise with letters 469/559 px, 4096x2048 noise with 469 px letters. Typical misreads: "GAATA...", "CAFTA".
- sdpa ViT attention gave the same answers as the reference-exact eager attention (gate 4096: identical output), at
  0.2 s vs 21 s vision for 16.7 MP. The official SGLang runtime misreads the gate image the same way.
- **Conclusion: model behaviour, not an engine bug.** The model loses giant glyphs in large busy images, the same in
  the official runtime. Realistic content (documents, UI, text on plain backgrounds) works at 4096 px, so there is **no
  size cap** (`INK_IMG_MAX_SIDE` stays opt-in) and **sdpa is the default ViT attention** (`INK_VIT_ATTN=eager` for
  reference checks).

## 7. Vision speed (2026-11-23, CPU work; GPU numbers pending tests/window_vision.sh)
Images only cost time on turn 1 (later turns restore the image tiles from the prefix cache), so this is turn-1 TTFT.

**Cost model** (sizes = square images; S = ViT length = 4 x tiles + 1; 23 layers, D 1024, MLP 4096):

| | 1 MP (1024^2) | 4 MP (2048^2) | 16.7 MP (4096^2) |
|---|---|---|---|
| tiles / S | 676 / 2,705 | 2,704 / 10,817 | 10,609 / 42,437 |
| linears (qkv, out, MLP) | 1.57 TFLOP | 6.26 TFLOP | 24.6 TFLOP |
| attention (global, 4096 S^2 per layer) | 0.69 TFLOP | 11.0 TFLOP | 170 TFLOP |
| patch embed / projector | 0.007 / 0.035 TFLOP | 0.03 / 0.14 | 0.10 / 0.56 |
| GPU estimate (GEMM ~170, sdpa ~120 TFLOP/s, glue ~1.5 TB/s) | ~20 ms | ~0.15 s | ~1.6 s (attention ~1.4 s) |
| host: JPEG decode (Pillow, libjpeg-turbo) | 8 ms | 19 ms | 71 ms |
| host: preprocess (to_rgb + bf16 canvas + tile copy), measured | 0.08 s | 0.65-1.5 s | 1.8-7.4 s |
| host -> GPU copy (bf16 tiles, pageable) | ~1 ms | ~3 ms | ~10 ms (102 MB) |

Host numbers are CPU measurements on the shared box (load average 17-46 on 12 cores, 2 torch threads); the bf16 band
fill (2.4 s at 16.7 MP) and the tile permute (0.33 s) were most of it. GPU numbers are estimates: the 2026-11-21 grid
timed vision without a sync (sdpa "0.2 s" is launch time only); gate 2026-11-22 had TTFT 2.9 s for the 16.7 MP image
including host work and a ~10.7k-token text prefill. Attention is ~87% of the 16.7 MP ViT FLOPs and is global over
the whole image (no window structure to exploit); linears dominate only below ~1.5 MP, where everything is small.

**Flags (vision.py, all off by default):**
- `INK_VIT_GPU_PRE=1` (exact): preprocess returns the raw uint8 pixels; Vision pads, normalizes (256-entry table
  made by the host formula) and tiles them on the device. Tiles and embeddings bit-identical (tests/test_vision.py).
  Host decode + preprocess measured: 16.7 MP 7.4 -> 0.30 s, 4 MP 1.48 -> 0.024 s, 1 MP 0.084 -> 0.007 s (loaded box).
  Half the bytes copied to the GPU. Not pinned: a pinned alloc in a handler thread could collide with a CUDA graph capture.
- `INK_VIT_ATTN=flash|cudnn|efficient` (sdpa pinned to one backend) and `=triton` (vit_attn.py: one flash kernel,
  bf16 QK^T / fp32 online softmax / bf16 P@V, reads vision.py's strided q/k/v views, writes [S,H,64] so the reshape is
  free; `INK_VIT_TRITON_CFG`, default 128,64,8,3, the only spill-free config in the sm_120 compile). Noise-gated.
- `INK_VIT_FP8=1`: ViT + projector linears through fp8_dense.mm (e4m3 weights per row, activations per token) in
  8192-row blocks. CPU real weights (70 tiles): embedding rel err 15% vs sdpa next to 3% eager-vs-sdpa noise, so it
  probably fails the gate; at most ~5% of the 16.7 MP time to win. Kept to measure.
- Skipped: weight-only FP8 (the ViT is compute-bound at large S; weights are 0.6 GB read once per layer pass), fused
  norm/RoPE/GELU glue (estimated ~45 ms at 16.7 MP, ~3%), tile batching (one image is already one sequence; there are no
  per-tile loops), FP8 attention (QK^T in e4m3 failed for the text model; FP8 P@V wasn't faster there).
