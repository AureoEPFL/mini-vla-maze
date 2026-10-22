# Methods and experimental protocol

## Abstract

Mini-VLA Maze studies the difference between recognising an instructed object and completing a sequence of contact actions. We train a compact network from synthetic expert demonstrations, with shared visual features for language grounding, semantic mapping and action prediction. We compare a direct imitation policy with a hybrid controller that applies explicit planning to the learned scene representation. The benchmark adds obstacles, timing and held-out layouts to an earlier continuous pushing baseline. Results are reported at episode level, with success and failure visualisations; exact measurements are in `../RESULTS.md`.

## 1. Task and simulator

A scene is an 11 x 11 grid rendered as an 88 x 88 RGB image. It contains one pusher, four coloured objects, an exit, static walls and optionally a moving mechanism. Objects have one of four colours and one of two shapes. The instruction identifies one colour/shape pair. Every generated scene includes distractors sharing its colour and its shape.

The action set is {up, right, down, left, wait}. A move changes the pusher position by one cell. Contact with the requested object pushes it one cell if the destination is free. Contacts with other objects are task failures; these protected distractors are not movable. Walls and active mechanism cells reject a move. A rejected move still advances time if the mechanism's interlock permits it. There is no inertia, torque control, continuous collision detection or friction model.

Moving mechanisms use a six-phase clock. A transition into a phase that would overlap the pusher or any object is paused by a safety interlock. Consequently the motion is interactive, but only through this explicit pause rule. Every phase transition uses the same function in the environment and planner.

Success means the instructed object reaches the exit without touching a protected distractor. Episodes have a 100-step limit. The renderer contains no marker identifying which object is the target. The tower is a stylised held-out geometry, not a realistic architectural map.

## 2. Demonstrations and splits

An A* teacher searches states `(pusher cell, target cell, obstacle phase)` with unit action cost. Its heuristic is the target's Manhattan distance to the exit. Other object locations remain fixed. The search has a 45,000-state budget and returns an explicit failure when exhausted or unsolvable.

Training uses 320 accepted episodes from seed 10,000 onward, cycling open, baffles, gate and sweeper. Validation uses 40 accepted episodes from seed 100,000 onward. Demonstrations of length 4 to 100 are retained. At each expert state, training stores RGB, grammar tokens, phase, semantic labels, target cell and the next four actions; the final chunk is padded with wait actions.

Test generation starts at 200,000, with a separate 10,000-seed block per suite. Each suite retains 16 oracle-solvable episodes of length 4 to 90. Acceptance and rejection seeds are saved. This conditions the benchmark on solvability and excludes difficult instances beyond the planner budget; it is not an estimate over every randomly sampled puzzle.

The blue disk is excluded as a commanded training and validation target. It can still appear as a distractor, and its pixels participate in semantic and attribute supervision. The experiment tests recombination into a held-out command, not recognition of a visually unseen object. Rotor and tower geometry are absent from training. Their rules are still known to the hybrid planner.

## 3. Network

The image is divided into 8 x 8 patches by a convolution with kernel and stride 8, producing 121 cell-aligned vectors of dimension 32. A 1 x 1 convolution refines each vector. This alignment is a strong prior: it assumes a fixed, registered camera and known cell size.

The fixed English instruction grammar is parsed into two IDs. Learned colour and shape embeddings have dimension 16. Independent linear projections produce colour and shape keys from each image feature. For cell i:

```text
c_i = dot(q_colour, k_colour,i) / sqrt(16)
s_i = dot(q_shape,  k_shape,i)  / sqrt(16)
alpha_i = softmax_i(c_i + s_i)
```

Adding the two logits encourages selection of their attribute intersection. Auxiliary labels directly supervise colour, shape and joint grounding. Attention shows the model's selection mechanism; it does not prove unrestricted language comprehension or provide a causal explanation by itself.

A 13-class per-cell semantic head recognises floor, wall, exit, pusher, eight object identities and moving-obstacle occupancy. A separate spatial stack receives visual features and the target attention map. After flattening, it is concatenated with both embeddings and sine/cosine obstacle phase. An MLP outputs 4 x 5 action logits.

There is no pretrained encoder, language model, transformer decoder or RL objective. The action chunk is a short sequence predicted from one observation, not a full collision-free trajectory.

## 4. Objective and optimisation

```text
L = CE(action chunk, expert chunk)
  + CE(semantic map, simulator labels)
  + CE(target cell, simulator target)
  + 0.4 * [CE(colour distribution) + CE(shape distribution)]
```

Colour/shape distributions are uniform over visible matching objects. Exact simulator state is used to generate supervision. It is not an input to the learned policy.

AdamW uses learning rate 7e-4, weight decay 1e-4, batch size 64 and gradient clipping at norm 2. Images receive brightness scaling in [0.8, 1.1] and Gaussian noise with standard deviation 0.01. Training runs 1,800 updates with seed 17. Every 100 updates, checkpoint selection minimises validation action CE plus twice each target and map error. The selected checkpoint is restored before evaluation. No test score enters checkpoint selection.

## 5. Controllers and information boundaries

| Controller | Learned components | Non-learned components / information |
|---|---|---|
| Direct k=1 | Visual encoder, grounding, action head | Structured parser and measured phase; observe after every action |
| Direct k=4 | Same checkpoint | Execute four actions before observing again |
| Hybrid | Same encoder, grounding, semantic head | Known grid dynamics and mechanism type/schedule; A* over predicted entities |
| Oracle | None | Ground-truth walls, positions, target and phase; A* |

The hybrid chooses its target by learned attention, not by querying the simulator's target ID. It decodes walls, pusher and object cells from its semantic head. The exit is remembered from visible observations, as an object can occlude it. It predicts the next symbolic state after an action and caches a route only while observed state agrees. Mismatches trigger replanning. Invalid perception returns wait; there is no oracle fallback.

The hybrid's success is **not** evidence that the neural action head learned long-horizon planning. The neural and hybrid comparisons isolate the practical value of explicit search on this constrained problem.

## 6. Evaluation

Main suites cover four familiar layout families, two unseen layout families and held-out blue-disk commands. Additional tests alter the instruction, blank/darken/noise the camera or move the pusher at step six. Wrong-instruction tests remain scored against the original task, so they are sensitivity probes; a positive language-following illustration instead changes the task itself on the same initial image.

Report episode success, Wilson 95% binomial intervals, steps, pushes, waits, blocked moves, repeated states and network/planner calls. Failure categories distinguish timeout from distractor contact. All test controllers use the same accepted scenes. A displacement oracle is included to expose intrinsic recovery limits.

The intervals describe variation across these episodes conditional on a single trained checkpoint. They do not measure variation across training seeds. Paired tests and multiple independently trained seeds would be required for stronger statistical conclusions. Network-call counts include failed episodes and are not a hardware latency benchmark.

## 7. Relationship to the continuous baseline

The earlier baseline maps RGB, a seven-word French instruction and pusher coordinates to four continuous 2D actions. Separate colour and shape compatibility scores ground the target. Its teacher approaches the correct side and pushes horizontally; vertical contact, arbitrary wall collisions and realistic dynamics are absent.

The maze version changes both dynamics and action representation to make multi-directional pushing and exact planning inspectable. It does not constitute a controlled ablation of only the original model architecture. The original CPU checkpoint and measured results remain available as a separate experiment; the new model is trained from scratch.

## 8. Limitations and next experiments

The main limitations are fixed camera registration, symbolic contact physics, simple object appearance, structured language, known mechanism dynamics and a small number of layouts. An RGB model on this renderer does not establish transfer to real cameras or robots.

Useful next experiments include independently repeated training seeds, more procedurally varied layouts, DAgger recovery data, temporal image inputs when phase is not directly observed, learned dynamics and a continuous-control simulator. Add these individually and retain the same evaluation protocol so their effects can be measured.

## References

- Zhao et al. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware / ACT](https://tonyzhaozh.github.io/aloha/).
- Ross, Gordon and Bagnell. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), 2011.
- Open X-Embodiment Collaboration. [Robotic Learning Datasets and RT-X Models](https://robotics-transformer-x.github.io/).

## 9. Larger arenas and constrained size transfer

The extension uses 25×25 grids, rendered at 200×200 pixels with the same 8×8 tiles. The patch encoder, semantic head and factorised attention operate convolutionally at this size. The action MLP expects the original 11×11 feature grid and is **not used** on larger maps. We report the hybrid and oracle only.

Three arena families are evaluated: alternating long walls, a spanning tree of nine rooms, and alternating walls with three timed airlocks. The airlocks use an eight-phase clock with offsets 0, 2 and 4; each closes for four phases. Door geometry, schedule and current phase are supplied to the planner, not estimated by the neural model. Walls, pusher, objects, exit and instructed target come from predicted visual outputs.

A* uses unit simulator-action costs and a relaxed object-distance heuristic computed by reverse flood fill. The search budget is 350,000 states. Six accepted scenes per family are selected from seed blocks 800000, 810000 and 820000, retaining oracle plans of 40–400 steps. The episode horizon is 450. Rejected seeds are recorded. Seeds are scene variations, not necessarily distinct wall topologies.

No retraining or large-map model selection is performed. Same tile appearance, grid registration, shared pushing rules and known mechanisms make this a constrained demonstration of spatial scalability, not generalisation to arbitrary visual environments or unknown dynamics. In Catacombs the requested object starts in a fixed room; this suite does not independently establish language dependence. The small-scene counterfactual tests address that separately.

The separate Colab run evaluated the identical checkpoint on CPU. Source, checkpoint, per-episode results and platform provenance were retained. Timing values are platform-dependent.
