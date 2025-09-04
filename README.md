# Graph-Augmented Neural Cellular Automata

We build on the classic differentiable Neural Cellular Automata (NCA) and augment it with a **mid-range, gated graph message passing** mechanism. The graph path exchanges information between cells beyond the 3×3 local neighborhood while being safely bounded and gated per channel. Training mixes short/long rollouts, stochastic fire rates, a **stability phase** against drift, and a **damage curriculum** for self-repair. The result is an NCA that can grow, maintain, and regenerate complex RGBA patterns more robustly.

---

## Scripts (one-liners)

### Modules
- `src/modules/ncagraph.py` — Hybrid Graph-Augmented NCA core: fixed Sobel perception, 1×1 MLP update, GroupNorm-on-dx, alive gating, and bounded graph residual.
- `src/modules/graph_augmentation.py` — Mid-range, sparsified attention over spatial offsets with channel-wise gating and (optional) zero-padded shifts.
- `src/modules/perception.py` — Frozen depthwise “identity + Sobel” stack that expands each channel to identity/∂x/∂y features.
- `src/modules/nca.py` — Classic NCA baseline: local perception + 1×1 MLP update with bounded step and alive gating (no graph path).

### Training
- `src/training/train_graph_ncagraph.py` — Full trainer for Graph-Augmented NCA with pool sampling, mixed rollouts, stability phase, damage curriculum, TB logging, and robust checkpoints.
- `src/training/train_intermediate_loss.py` — Classic NCA trainer using TARGET-masked MSE + tiny alpha-area penalty with short/long rollouts and optional stability.

### Testing / Analysis
- `src/testing/test_graph_augmented_nca.py` — Diagnostic rollout that saves RGB/attention panels and offset tiles to inspect graph behavior step-by-step.
- `src/testing/test_graph_augmented_regeneration.py` — Regrowth/repair tests: applies specific damage kinds at a chosen step and writes PNG/MP4 sequences.
- `src/testing/test_intermediate_loss.py` — Classic NCA growth demo from a seed; saves frame grids and optional side-by-side with the target.

### Utils
- `src/utils/config.py` — Loads `configs/config.json` and exposes typed accessors for training/testing.
- `src/utils/image.py` — Reads RGBA targets (PNG), normalizes to [0,1], and returns tensors shaped for the CA.
- `src/training/pool.py` — Simple replay/pool of CA states with sample/replace to maintain diversity.
- `src/utils/visualize.py` — Side-by-side target vs. prediction compositing and image saving helpers.
- `src/utils/utility_functions.py` — Small helpers: parameter counting, formatting, and misc utilities.
- `src/utils/damage.py` — In-place damage operators and a unified `apply_damage_policy_` driven by the config (curriculum + mixture of kinds).

### Configs & Data
- `configs/config.json` — Central configuration for data, model, training, logging, graph, and damage knobs.
- `data/emojis/` — RGBA targets used for training and evaluation (e.g., `gecko.png`, `heart.png`).

### Outputs
- `outputs/graphaug_nca/...` — Checkpoints, logs, images, and videos for the graph-augmented runs.
- `outputs/classic_nca/...` — Checkpoints, logs, images, and videos for the classic NCA baseline.