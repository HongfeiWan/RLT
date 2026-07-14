# RLT Online RL Runtime

This README describes the `rlt_online_rl` runtime: the lightweight online RL
system used after a frozen VLA/RL-token model has been trained and served. The
root [README](../README.md) defines GR00T as this checkout's primary backend;
the historical openpi path remains available for the Agilex experiment.

`rlt_online_rl` implements the robot-facing online learning loop:

- a Machine A feature/reference service that serves `z_rl` and VLA reference
  chunks
- a Machine B actor service, learner service, and replay manager
- a robot rollout driver that connects ROS observations, Machine A, the actor,
  replay, manual signals, reset, and evaluation

The validated legacy task configuration is `configs/tasks/agilex_ethernet`,
which demonstrates Ethernet insertion on a real robot. The GR00T/Nero file at
`configs/tasks/groot_nero/online_rl.example.yaml` is intentionally a
fail-closed starting template, not a validated robot configuration.

## Scope

This package owns the online RL runtime:

- B1 `actor_service`
- B2 `learner_service`
- B3 `replay_manager`
- B4 `EnvDriver`
- ROS rollout adapter
- replay journal, raw episode persistence, actor snapshots, logs, and metrics
- warmup, online rollout, human takeover, critical-phase handoff, and eval-only
  execution

It does not train the base VLA or the RL-token module. The primary GR00T tools
live in `groot_rlt`; the retained openpi tools live in `src/openpi` and
`scripts/train_rlt.py`.

## Runtime Architecture

Machine A runs either the primary GR00T/RLT feature server or the retained
openpi/RLT policy server. For each observation it returns:

- `z_rl`: the compact RL-token feature
- `ref_chunk`: the VLA reference action chunk
- `proprio`: optionally supplied by GR00T for native nested state observations

Machine B runs:

- `actor_service`: serves the current lightweight actor for low-latency
  refinement
- `learner_service`: samples replay, trains actor/critic, and publishes actor
  snapshots
- `replay_manager`: owns the replay buffer and append-only journal

Robot rollout:

- reads ROS observations
- queries Machine A
- executes either the VLA reference chunk or the actor-refined chunk
- records raw step traces
- builds replay transitions at episode end
- sends transitions to the replay manager

## Chunk Execution Path

At each chunk boundary:

1. The rollout adapter reads the current robot observation.
2. The observation is sent to Machine A.
3. Machine A returns `z_rl`, `ref_chunk`, and optionally `proprio`.
4. The rollout uses server `proprio` for nested GR00T state, otherwise it
   derives it from the local flat observation state.
5. During warmup or non-critical full-task prefixes, the robot executes
   `ref_chunk` directly.
6. During online critical-phase control, Machine B actor receives
   `z_rl / proprio / ref_chunk` and returns a refined chunk.
7. The robot executes the selected chunk for `chunk_exec_horizon` control ticks.
8. The episode stores raw executed steps first.
9. At episode end, replay windows are built and any missing Machine A anchors
   are backfilled.
10. The learner samples replay and publishes actor snapshots.
11. The actor service hot-loads the latest snapshot.

## Core Modes

### Warmup

Warmup collects replay using the frozen VLA reference policy. The actor is not
allowed to control the robot yet. The learner starts once replay reaches
`warmup_min_size`.

If `warmup_post_collect_updates` is set, the learner performs that many warmup
updates before online rollout is allowed. Otherwise, the required warmup update
budget is derived from `warmup_ready_adds_total * grad_updates_per_cycle`.

### Warmup Wait Online

After enough warmup data has been collected, rollout waits for both:

- learner status `ready_for_online == true`
- actor version at or above the rollout threshold

The switch to online control happens only between episodes, never in the middle
of an episode.

### Online

Online episodes can use the actor in the critical phase. The actor is stochastic
or deterministic according to `runtime.env_driver.actor_deterministic` during
training. Eval-only rollout forces deterministic actor mean.

### Critical Phase vs Full Task

`critical_phase` starts the episode directly in the precision-critical segment.
`full_task` uses the base policy before the critical segment, then switches to
critical control after the manual critical-phase signal. Non-critical full-task
prefixes are not written to replay.

## Current Ethernet Defaults

The current Ethernet task config uses:

- `action_dim: 7`
- `chunk_len: 10`
- `z_dim: 2048`
- `proprio_dim: 7`
- `action_representation: delta_chunk`
- `reference_dropout_prob: 0.5`
- `warmup_min_size: 600`
- `warmup_post_collect_updates: 20000`
- `grad_updates_per_cycle: 5`
- `step_trace_stride: 0`
- `control_frequency_hz: 20.0`

Note that `step_trace_stride: 0` disables dense stride replay and keeps
chunk-boundary replay. This is an intentional runtime setting for the current
Ethernet configuration.

## Replay Semantics

Each replay transition contains:

- `z_rl`, `proprio`
- `ref_chunk`: Machine A / VLA reference for the transition observation
- `action_chunk`: the action actually executed on the robot
- `rewards`, `done`
- `next_z_rl`, `next_proprio`, `next_ref_chunk`
- `source`: chunk-level control source
- `source_chunk`: per-step control source
- `collection_phase`: warmup or online
- `episode_id`, `step_id`, `success`, `intervention_flag`

`TransitionSource` is only a control-source label:

- `BASE`: frozen VLA reference execution
- `RL`: actor-refined execution
- `HUMAN`: human-controlled execution
- `MIXED`: a window containing both human and policy steps

The trainer uses `source_chunk` to choose the BC target step by step:

- `HUMAN / MIXED` steps align to the executed `action_chunk`
- `BASE / RL` steps align to the VLA `ref_chunk`

This is intentionally different from replacing `ref_chunk` with human actions.
At deployment time the actor still sees VLA references, so human data teaches
how to edit a VLA reference into the executed correction.

## Learner Objective

The learner uses a twin critic, a fixed-std Gaussian chunk actor, target
networks, and reference-action dropout. The current actor loss is:

```text
actor_loss = bc_weight * bc_penalty - q_weight * actor_q + delta_weight * delta_penalty
```

The warmup and online stages can use different BC/Q weights:

- `warmup_bc_weight`, `warmup_q_weight`
- `online_bc_weight`, `online_q_weight`

`delta_penalty` is computed after converting normalized training actions back to
executable absolute chunks. `delta_action_indices` selects the compared action
channels; when omitted it preserves the historical Agilex behavior and uses the
first `min(6, action_dim)` channels. Non-contiguous indices can be supplied for
GR00T/Nero layouts.

For Nero, generate the 26D file with
`groot-rlt export-online-stats`; do not point this field directly at GR00T's
grouped `statistics.json` or the raw 19D LeRobot action stats. The loader honors
the export's declared `lower_key`/`upper_key`, including symmetric `min`/`max`
bounds, and can verify `action_layout_hash`/`proprio_layout_hash` before use.

Machine-A payloads may include a flat `proprio` vector. The runtime uses it only
for GR00T's nested observation-state layout; a legacy flat `observation["state"]`
remains authoritative even when the payload also contains `proprio`.

## Replay Windows

Replay is built from raw episode traces at episode end.

`step_trace_stride: 0`:

- uses chunk-boundary replay windows
- adds policy-restart anchors when human control returns to policy control
- may add one terminal-aligned final window
- backfills only the anchors needed by those windows

`step_trace_stride > 0`:

- builds dense replay windows from raw step traces at the configured stride
- uses batched Machine A feature backfill for missing anchors
- matches the dense replay idea used by the RLT paper when set to `2`

The replay journal is an append-only pickle stream. The replay manager restores
from it on startup and continues episode numbering from the largest restored
`episode_id + 1`.

## Human Control And Manual Signals

The ROS adapter supports manual services for:

- requesting the next episode
- recording success, failure, or done
- entering or toggling the critical phase
- selecting whether the next critical phase uses actor or base policy
- toggling teleop takeover

During human takeover, replay records the latest human action sampled at each
control tick. It does not insert the raw teleop event stream directly into
replay.

## Installation

Use a separate Python 3.10 environment for the online RL runtime:

```bash
export RLT_ROOT=/home/whf/Project/RLT
cd "$RLT_ROOT/rlt_online_rl"
conda create -y -n rlt_online_rl310 python=3.10 pip
conda activate rlt_online_rl310
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ../packages/openpi-client
python -m pip install -e .
```

Robot rollout and keyboard clients also need ROS sourced in the shell where they
run:

```bash
source /opt/ros/humble/setup.bash
```

Optional W&B sidecar support:

```bash
python -m pip install -e '.[monitor]'
```

## Launching Training

Start the Machine B services with an explicit task config. For GR00T/Nero,
first replace every `REPLACE_*` value in the example; it is designed to fail
before training when statistics/layout metadata are still missing:

```bash
cd "$RLT_ROOT/rlt_online_rl"
conda activate rlt_online_rl310
python launch/launch_machine_b.py \
  --config configs/tasks/groot_nero/online_rl.example.yaml
```

Start the primary GR00T Machine A in the Isaac-GR00T environment:

```bash
cd /home/whf/Project/Isaac-GR00T
.venv/bin/groot-rlt serve-features \
  --groot-repo-path /home/whf/Project/Isaac-GR00T \
  --model-path <groot-checkpoint> \
  --processor-path <processor-directory> \
  --vlm-model-path <Cosmos-Reason2-2B> \
  --rl-token-checkpoint <rl-token-checkpoint.pt> \
  --action-dim 26 \
  --proprio-dim 26 \
  --chunk-len 10 \
  --denoise-steps 32 \
  --port 8000
```

The retained openpi Machine A is still available for the legacy Agilex task:

```bash
cd "$RLT_ROOT"
python scripts/serve_rlt_policy.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --checkpoint-dir <checkpoint-dir> \
  --port 8000 \
  --shared-prefix-inference
```

`--shared-prefix-inference` is an inference-only latency optimization for the
Machine A server. Machine A needs both `z_rl` and the frozen VLA `ref_chunk`;
the legacy path computes the VLA prefix once for `z_rl` and again inside action
sampling to build the KV cache. With this flag, the server computes the prefix
once and reuses the same prefix output/KV cache for both outputs. It does not
change training, checkpoints, model weights, normalization, or the online RL
runtime payload. Omit the flag to keep the legacy inference path for strict
reproduction.

For local integration tests without the real VLA server:

```bash
cd "$RLT_ROOT/rlt_online_rl"
python launch/fake_machine_a.py
```

The existing robot rollout below is the legacy 7D Pika/Agilex ROS adapter. It
must not be used with the 26D GR00T template. GR00T rollout instead requires an
explicit target-specific `env_factory`; that interface is intentionally left
unselected until the robot/Teleop action contract is validated.

Legacy robot rollout:

```bash
cd "$RLT_ROOT/rlt_online_rl"
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310
python launch/launch_robot_rollout.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://MACHINE_A_IP:8000
```

Start the training keyboard client:

```bash
python keyboard_toggle_teleop_record_reward_isolation.py
```

## Launching Eval

Eval does not start learner or replay. It runs actor inference and robot rollout
only.

Start an actor service:

```bash
cd "$RLT_ROOT/rlt_online_rl"
conda activate rlt_online_rl310
python scripts/run_online_rl.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --system.role actor_service \
  --system.actor_service.snapshot_path <actor_snapshot.pkl>
```

Start eval rollout:

```bash
python launch/launch_actor_eval.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://MACHINE_A_IP:8000
```

`launch_actor_eval.py` waits for the actor service and then starts
`pika_sync_ros.py --eval_actor_only`. Eval-only rollout forces deterministic
actor mean.

Start the eval keyboard client:

```bash
python keyboard_actor_eval.py
```

## Common Tools

Inspect replay:

```bash
python scripts/tools/inspect_replay_journal.py \
  runs/agilex_ethernet/replay/replay_journal.pkl
```

Plot learner metrics:

```bash
python scripts/tools/plot_learner_metrics.py \
  --run_dir runs/agilex_ethernet
```

Offline training and analysis tools are documented in
[scripts/offline/README.md](scripts/offline/README.md).

Real-robot replay export and playback tools are documented in
[scripts/replay_real_robot/README.md](scripts/replay_real_robot/README.md).

## Suggested First Run Order

Training:

1. Start Machine B services.
2. Start Machine A.
3. Start robot rollout.
4. Confirm the robot has reset to the start pose.
5. Start the training keyboard client.
6. Press `o` to begin an episode.
7. In `full_task`, press `c` at the critical-phase boundary.
8. Press `s` for success or `f` for failure.

Eval:

1. Start actor service.
2. Start eval rollout.
3. Confirm the robot has reset to the start pose.
4. Start the eval keyboard client.
5. Press `a` or `b` to select actor or base for the next critical phase.
6. Press `o` to begin an episode.
7. In `full_task`, press `c` at the critical-phase boundary.
8. Press `s` when the episode should end.

## Common Misunderstandings

- `full_task` prefixes do not write replay until the critical phase starts.
- `full_task` prefixes do not use the actor; they execute the Machine A
  reference.
- In training, `s` means success and done.
- In eval, `s` ends/resets the episode and is not used as a training reward.
- Eval-only rollout ignores stochastic training rollout settings and uses actor
  mean.
- `a / b` in eval select the next episode's critical policy, not an immediate
  mid-episode switch.
- `critical_phase` usually does not need `c`, because it starts already inside
  the critical segment.
- Warmup readiness never switches the current episode to online control halfway
  through the episode.

## Directory Map

```text
rlt_online_rl/
|-- configs/                    # Base and task runtime configs
|-- launch/                     # Machine B, rollout, eval, and fake Machine A launchers
|-- scripts/offline/            # Offline replay training and analysis
|-- scripts/replay_real_robot/  # Export and playback reference/actor joint chunks
|-- scripts/tools/              # Lightweight inspection and plotting tools
|-- src/rlt_online_rl/          # Core runtime package
|-- train_deploy_alignment/     # ROS adapter and manual signal bridge
`-- tests/                      # Runtime unit tests
```
