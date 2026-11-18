# Action-conditioned world model: stage-0 feasibility

Fully observed, noiseless linear dynamics; least-squares baselines. This is intentionally easy and not a novel visual world model.
New diagonal action combinations are held out from training; this does not establish nonlinear or visual OOD generalization.

| Method | Data seed | Position RMSE H10 | Position RMSE H30 |
|---|---:|---:|---:|
| action_conditioned | 7 | 4.80959e-15 | 2.38508e-14 |
| action_blind | 7 | 0.583957 | 2.05159 |
| action_conditioned | 17 | 3.35946e-15 | 1.6269e-14 |
| action_blind | 17 | 0.586029 | 2.06177 |
| action_conditioned | 27 | 2.21831e-14 | 1.22191e-13 |
| action_blind | 27 | 0.58598 | 2.07494 |

| Planning model | Successes / same64 cases | Median final distance |
|---|---:|---:|
| action_conditioned_seed7 | 64/64 | 0.00892864 |
| action_blind_seed7 | 0/64 | 1.19274 |
| action_conditioned_seed17 | 64/64 | 0.00892864 |
| action_blind_seed17 | 0/64 | 1.19274 |
| action_conditioned_seed27 | 64/64 | 0.00892864 |
| action_blind_seed27 | 0/64 | 1.19274 |
| oracle_known_dynamics | 64/64 | 0.00892864 |

The64 cases are reused across models;448 runs are not448 independent cases. Success means final distance<=0.2 after32 controls.
The action-blind planner has B=0 by construction. This demonstrates the need for an action interface, not superiority over competing modern world models.
Oracle uses known dynamics and is not learned. No P values or real-task population confidence intervals are reported.
Only state/action system identification and closed-loop wiring are demonstrated. Pixels, occlusion, contact, partial observability and changed dynamics are untested.
Next step is a separately designed pixel/latent model study with multiple dynamical worlds, matched compute and episode-level splits; retain oracle/blind controls.

Reproduce with `python research/action_world_stage0_20261114/experiment.py --output NEW_DIRECTORY`.
