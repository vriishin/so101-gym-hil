# Milestone 5 visual-policy report

This report summarizes an executed local experiment. Generated datasets, replay buffers, and model
weights are intentionally excluded from the source repository; paths below identify the original run
layout and hashes identify the selected artifacts.

## Frozen Milestone 4 teacher

- Frozen local checkpoint: `artifacts/milestone4/selected_state_policy.zip` (not committed)
- SHA-256: `0F072091FAC296C8F54987EA5D37C8DD42FBA0EA1FBDC59B67AF6B0F4C5C9A65`
- Original experiment code baseline: `46ec8dc8a53ddec085f0ee77e245eebbd343177d`
- Resolved configuration: `outputs/full_state_phase5d/resolved_config.yaml`
- Final report: `outputs/full_state_phase5d/final_evaluation.json`
- Evaluation seeds: 18000 through 18019
- Privileged teacher schema: 42 values from `full_task_state_vector`
- Reloaded full-task success: 100%; random full-task success: 0%

## Visual demonstrations and policy contract

The executed recorder produced 100 successful trajectories in 100 attempts, totaling 6,948
transitions. Every transition contains 64x64 front RGB, 64x64 wrist RGB, 12 robot proprioceptive
values, the requested six-dimensional policy action, the actually executed six-dimensional action,
reward, success metadata, and task-stage labels. Stage and success fields are diagnostic-only and are
never returned by `VisualObservationAdapter`.

The policy key set is exactly:

- `observation.images.front`: uint8 `[64, 64, 3]`
- `observation.images.wrist`: uint8 `[64, 64, 3]`
- `observation.state`: float32 `[12]`, six joint positions followed by six joint velocities

The adapter rejects extra keys and specifically tests cube, target, TCP, contact, and phase leakage.
Both camera streams were nonblank in the executed smoke test.

## Behavior cloning

The selected dual-camera encoder checkpoint is
`outputs/visual_milestone5/behavior_cloning/best.pt` (SHA-256
`05207CAB97D5F526E1E95F8C51EBE674137654000C55A9445A8527A48F929D02`). It was trained for 30 epochs
on 80 episodes and validated on 20 held-out episodes. The gripper output received a fourfold loss
weight after the first run diagnosed premature reversal immediately below the release threshold; no
phase or privileged feature was added.

| Phase | MAE | RMSE |
|---|---:|---:|
| grasp | 0.0792 | 0.1165 |
| lift | 0.1095 | 0.1417 |
| transport | 0.0618 | 0.0956 |
| descent | 0.0992 | 0.1548 |
| release | 0.0622 | 0.1241 |
| retreat | 0.0725 | 0.1064 |
| stability | 0.1165 | 0.1591 |
| all | 0.0789 | 0.1222 |

Closed-loop evaluation over seeds 18800 through 18819 achieved 100% reach, grasp, lift, and target
entry; 35% release and stable success; and 0% drops. Reloading the checkpoint reproduced the same
result.

## SAC/RLPD diagnostic

The local adapter uses the behavior-cloned encoders, fresh latent-space twin critics and targets, a
fresh bounded online replay, and 50% demonstration samples per update. The external action remains
six-dimensional and all structured phase-aware environment mappings are unchanged. LeRobot 0.6.0
was inspected, but its runtime is not installed here; this is the documented compatible-adapter path.

- 1k pipeline: 1,000 online transitions, 250 optimizer updates, selected reload 30% stable success.
- 10k diagnostic: 10,000 online transitions, 2,500 optimizer updates, ten evaluation checkpoints.
- Best 10k checkpoint:
  `outputs/visual_milestone5/rlpd/diagnostic_10k/checkpoints/best.pt`, SHA-256
  `81449D251E57A982A2C9B64C4E6E393D28A8E73A787717E34D38DC0D8D46FD29`.
- Selected checkpoint is a separately saved copy of that evaluated best model at
  `outputs/visual_milestone5/rlpd/diagnostic_10k/checkpoints/selected.pt`.
- Final diagnostic checkpoint SHA-256:
  `A62FF2600193956525E035E87F0FAA7C2D54C21646891D44605D01050C729003`.
- Independently reloaded selected result on seeds 19000 through 19009: 100% reach, grasp, lift, and
  target entry; 30% release and stable success; 0% drops.

The online policy fell to zero release/success at 2k and did not recover, although all preceding stages
remained at 100%. This is release-boundary actor drift, not an image, reach, grasp, lift, transport,
replay, or optimizer failure. Because the stage metrics did not improve, the run was not extended.
The initial/best, every evaluation, and final checkpoints are retained separately.

Milestone 5 passes its smoke gate because visual-only training, checkpointing, reload, and evaluation
all executed and the independently reloaded selected visual policy has non-zero full-task success.
This is not evidence that visual SAC fine-tuning improved over behavior cloning, and no reward-study
claim is made.
