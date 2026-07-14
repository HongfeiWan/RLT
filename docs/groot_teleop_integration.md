# GR00T RLT / Teleop integration contract

This note records the read-only interface review of `zhangbt@node0:~/Teleop`
(the actual directory name is `~/Teleop`) and defines the boundary for a future
live Groot-RLT adapter. It is deliberately not a real-robot launch guide.

## Decision

Do not connect the current 26D RLT action directly to the Teleop hardware path.
Keep the existing Pika/Agilex ROS adapter unchanged and add a separate Teleop
policy adapter only after the action-space mismatch below is resolved.

Teleop already owns the safety-critical parts of the loop:

- operator input and human/policy authority;
- safety hold and guarded rollout;
- RTC/history stitching;
- command timing and robot output;
- action, policy, authority, and episode capture.

Groot-RLT should therefore integrate as a policy provider. It must not bypass
Teleop's authority mux or call the CAN/robot drivers directly.

## Interfaces verified on node0

The following paths and contracts were inspected without running the robot:

| Teleop path | Contract |
| --- | --- |
| `src/teleop_stack/policies/base.py` | `PolicyObservation`, `PolicyActionChunk`, `PolicyInterface` |
| `src/teleop_stack/session/action_authority.py` | `ActionAuthorityMux`, `AuthorityDecision`, intervention IDs and control-source provenance |
| `src/teleop_stack/models.py` | `SingleArmTeleopCommand` and embodiment-neutral `CommandEnvelope` |
| `src/teleop_stack/policies/groot_policy.py` | native GR00T observation construction and action-to-command conversion |
| `src/teleop_stack/data_capture/exporters/rlt_episode.py` | offline export shaped like this repository's RLT episode sidecars |

The intended live shape is:

```text
PolicyObservation
  -> GrootRltTeleopPolicy.infer(...)
       -> Machine A: z_rl + physical GR00T reference + proprio
       -> Machine B: actor refinement
       -> denormalize exactly once
       -> validated 26D physical action converter
  -> PolicyActionChunk
  -> PolicyTrajectoryManager
  -> ActionAuthorityMux
  -> rollout guard / robot adapter
```

`ActionAuthorityMux` remains outside the RLT policy. Human takeover must cancel
or supersede policy authority there, while preserving the policy proposal and
reference action for replay provenance.

## Blocking action-space mismatch

The currently supported Groot-RLT/Nero layout is:

```text
eef_9d[9] + hand_joint_target[10] + arm_joint_target[7] = 26D
```

The inspected Teleop `action_dict_to_commands` path creates a
`SingleArmTeleopCommand` from `eef_9d` and `hand_joint_target`. In that path the
additional `arm_joint_target[7]` is not represented by the emitted command.

A direct adapter would therefore advertise and normalize 26 channels while the
hardware command consumes only 19 of them. This would invalidate actor/critic
training, layout hashes, reference regularization, and replay interpretation.
It must fail closed rather than silently project the chunk.

One of these designs must be chosen and reflected consistently in the GR00T
checkpoint, feature server, statistics, actor, replay, and command converter:

1. a true 26D command envelope whose robot adapter consumes all 26 channels;
2. a documented 19D deployment action space with a separately trained/exported
   19D checkpoint, statistics, layout hash, and RLT actor;
3. an explicit constrained projection with a defensible control meaning and
   training data generated in that projected space.

Dropping channels only at execution time is not an acceptable fourth option.

## Other unresolved live-runtime contracts

- Teleop's base `RobotInterface` currently exposes
  `connect/send_command/stop/disconnect`; it does not define the
  `reset/observe/step` contract expected by a chunk environment. Robot state is
  available through profile-specific trace snapshots, which should first be
  promoted to a stable observation adapter.
- Teleop records terminal success/failure but does not currently expose a
  numeric per-step reward contract. Until configured, RLT should use only an
  explicit sparse terminal label and must not invent dense rewards.
- The existing `rlt_online_rl` `human_override_factory` replaces a whole chunk.
  Teleop can take over in the middle of a chunk and carries intervention IDs,
  resume-gate state, and safety-hold state. A correct adapter must record the
  authority decision at each executed control tick instead of flattening it to
  one chunk-level Boolean.
- The reviewed Teleop policy runs near 10 Hz with a 32-step GR00T horizon and
  an 8-step replan convention, while the current RLT example uses a 50 Hz
  control loop and a 10-step chunk. The execution rate, replan interval,
  sample-and-hold/interpolation rule, and replay timestamps must be made one
  explicit contract.

## Future adapter protocol

The future adapter should structurally implement Teleop's `PolicyInterface`:

```python
class GrootRltTeleopPolicy:
    @property
    def policy_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    def reset(self, *, episode_id: str | None = None) -> None: ...

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk: ...
```

Its constructor/configuration must require, rather than guess:

- Machine-A URL and actor-service URL;
- action, proprio, RL-token, and chunk dimensions;
- ordered action and proprio layout hashes;
- action representation and versioned normalization-statistics file;
- camera key mapping and task-instruction source;
- command converter/profile name;
- request deadline and a fail-safe result on timeout;
- deterministic/evaluation mode;
- policy/checkpoint/version identifiers written to capture metadata.

Before returning a chunk it must validate:

- exact feature dimensions and finite values;
- exact layout-hash equality across server, stats, and adapter;
- physical versus normalized action space;
- horizon and action timestamps;
- every action channel is consumed exactly once by the command converter;
- the result passes Teleop's existing rollout guard.

RTC should remain in Teleop's execution/trajectory layer. The GR00T feature
server is intentionally stateless so replay features can be reconstructed out
of order.

## Provenance mapping

The live capture adapter must preserve these distinctions:

| Teleop authority result | RLT meaning |
| --- | --- |
| policy command executed unchanged | VLA or actor behavior, according to the selected policy source |
| human intervention command executed | human-intervention behavior; retain actor and VLA proposals separately |
| safety/hold command | safety hold, not an ordinary human correction |
| policy proposal during dry-run | proposal only; never mark it executed |
| terminal success/failure event | sparse terminal label with its label source |

`source_name` alone is not enough. Store the authority state, intervention ID,
executed command, actor proposal, VLA reference, timestamps, and layout/version
metadata at each decision boundary.

## Offline exporter caveat

Teleop's current `RltEpisodeExporter` is useful as a schema handoff, but its
exported RL-token tensors are empty and its VLA reference keys are unset. Such
an export is not a ready-to-train VLA warmup replay. It must first be enriched
with RL tokens and reference chunks from the exact frozen checkpoint, then
validated by `groot_rlt.episode_schema` before replay construction.

## Acceptance gates before enabling hardware

1. Unit-test observation mapping and the complete action-channel mapping.
2. Run recorded-episode replay with no hardware output.
3. Run Teleop `rollout_dry_run`; compare candidate, reference, and authority
   logs without sending commands.
4. Run guarded simulation/digital-twin rollout with forced takeover and timeout
   tests.
5. Verify normalization and layout hashes against captured physical actions.
6. Verify reset, reward, terminal label, intervention, resume, and safety-hold
   semantics.
7. Only then add an explicit hardware-enabled launch; it must remain opt-in.

Until all gates pass, the repository's existing robot/teleop launchers remain
the only implemented interfaces and no Groot-RLT command selects them by
default.
