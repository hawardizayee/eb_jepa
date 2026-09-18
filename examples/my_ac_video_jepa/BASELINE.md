# Baseline acceptance criteria — AC Video JEPA

*Pre-registered before the baseline run — written to fix the goalpost before the
result is known. Do not edit the criteria after seeing the run; record the outcome
in the annotated `baseline-ac-video` git tag instead.*

## Reference
- Source: **reported** in `examples/ac_video_jepa/README.md` (not re-run on this
  machine), "Results" table — Impala encoder + RNN predictor, MPPI planner,
  **Random Wall** setup.
- Original planning success rate = **97 ± 2%** over N=20 episodes.
- Setup match confirmed: `fix_wall: false` in
  `eb_jepa/datasets/two_rooms/data_config.yaml` randomizes both wall and door
  location — this *is* the Random Wall task. `level: normal` in `cfgs/eval.yaml`
  takes the general start/target path (`env.py:233` special-cases only `"easy"`).
- Caveat: the README's ±2 is the std over **3 training seeds × the 3 last epoch
  checkpoints**. This baseline runs **one** seed. The reported band therefore
  understates the noise on a single run and cannot be reused as-is — see
  Tolerance rationale.

## Reproduced if
- **Planning success rate ≥ 89%**, from `main_eval`'s `success_rate` over the
  `num_eval_episodes: 20` episodes in `cfgs/eval.yaml`. This is the real signal —
  not the training loss. One-sided on purpose (see rationale).
- **Unroll error degrades gracefully** — `val_rollout/mean_mse/{t}` from
  `main_unroll_eval` rises monotonically-ish with horizon `t` and stays bounded;
  no blow-up or step change. Compounding prediction error is expected; divergence
  is not.
- **No collapse** — `train/regl/std_loss` stays **near 0**, not near its ceiling of
  `std_margin = 1.0`. The loss is `mean(relu(1.0 - std))`, so 0 means every feature
  clears the margin and 1.0 means the embeddings have collapsed to a constant.
  `train/regl/idm_loss` should also keep decreasing. The README's ablation table is the
  canary here: `IDM coeff = 0 → SR 1 ± 1%`, and zeroing var or cov drops SR to
  ~46-47%. A collapsed run is unmistakable, not marginal.

## Fixed setup
- Seed: **1** (`meta.seed` in `cfgs/train.yaml`)
- Training config: `cfgs/train.yaml` — Impala encoder, RNN predictor, 12 epochs,
  batch 384, `nsteps: 8`, regularizer `cov=8 std=16 sim_t=12 idm=1`, `use_proj: false`
- Planner config: `cfgs/planning_mppi.yaml` — MPPI, `plan_length: 90`,
  `n_iters: 20`, `num_samples: 200`, `num_elites: 20`, `sum_all_diffs: true`
- Eval config: `cfgs/eval.yaml` — 20 episodes, `n_allowed_steps: 200`, `level: normal`
- Metric: planning success rate on the Random Wall task
- wandb: group `baseline-ac-video`, run tagged `baseline`, launched from a clean
  tree so the commit SHA is recorded

## Tolerance rationale
The floor is one-sided because success rate sits near its ceiling: beating 97% is
not a reproduction failure, so a symmetric band would only add a meaningless upper
bound.

Where 89% comes from: a single 20-episode eval at a true rate of 97% carries a
binomial std of `sqrt(0.97 × 0.03 / 20) ≈ 3.8%`. Two std below the reference is
≈ 89.4%. Rounding to **89%** gives a floor that absorbs single-run episode noise
plus hardware and library differences, while still sitting far above every failure
mode in the README's ablation table (the *mildest* ablation, dropping the time
similarity term, lands at 61%). The gap between 89% and 61% is what makes this
band able to separate "reproduced" from "broken" rather than merely "noisy."

Cheap way to tighten without 3 seeds: `save_every_n_epochs: 1` already writes a
checkpoint per epoch, so averaging success rate over the **last 3 epoch
checkpoints** mirrors half of the README's protocol at no extra training cost. Do
that before widening the band if the first result lands near the floor. Upgrade to
a multi-seed estimate (tolerance ≈ 2×std over 3 seeds) only if a Phase-B result
needs to be publishable.

## Baseline frozen (YYYY-MM-DD) — pointers, not criteria
*Fill in after the run. Do not edit anything above this line.*

- **Result:** PASS / FAIL — success rate __%, against the ≥89% floor.
- **Unroll:** `val_rollout/mean_mse/{0..3}` = __ / __ / __ / __
- **Collapse check:** final `train/regl/std_loss` = __, `train/regl/idm_loss` = __
- **Code anchor:** `git tag baseline-ac-video` → commit `_______`. Diff later work
  with `git diff baseline-ac-video -- examples/my_ac_video_jepa/`.
- **Metrics anchor:** wandb run `_______`, tagged `baseline` in group
  `baseline-ac-video`.
- **Weights:** run dir `_______` holding `latest.pth.tar` and `e-{0..11}.pth.tar`.
  Back these up outside git before tearing down the machine.
