# Experiment viewer

Open [gallery.html](gallery.html) in a browser to switch between runs. The selection includes successes, a neural-policy failure, and all three larger arenas. Every animation labels its controller and displays the raw model input beside the styled view.

![Three airlocks](media/viewer_airlocks.gif)

![Nine connected rooms](media/viewer_catacombs.gif)

The new viewer uses the first accepted test scene for each large arena. The earlier diagnostic gallery below preserves its original selection rule.

# Behaviour gallery

Every animation identifies the controller, instruction, seed and outcome. Success and failure examples use the **first matching episode** in the declared test order. Counterfactual illustrations use a separate seed set and are excluded from benchmark scores.

## Continuous action baseline

![Continuous neural policy](media/baseline_seen_k1_success.gif)

*The baseline predicts continuous action chunks directly; it has no route planner.*

![Baseline failure](media/baseline_seen_k1_failure.gif)

## Maze controllers

### baffles — direct — success

Instruction: `push the green disk to the exit`. Seed: 210009. 18 steps, 12 pushes, 0 waits, 0 blocked moves.

![direct baffles success](media/baffles_direct_success.gif)

### baffles — direct — timeout

Instruction: `push the amber disk to the exit`. Seed: 210016. 100 steps, 3 pushes, 0 waits, 96 blocked moves.

![direct baffles timeout](media/baffles_direct_timeout.gif)

### baffles — direct — distractor_contact

Instruction: `push the green square to the exit`. Seed: 210007. 2 steps, 0 pushes, 0 waits, 1 blocked moves.

![direct baffles distractor_contact](media/baffles_direct_distractor_contact.gif)

### gate — direct — success

Instruction: `push the green square to the exit`. Seed: 220010. 18 steps, 10 pushes, 0 waits, 0 blocked moves.

![direct gate success](media/gate_direct_success.gif)

### gate — direct — timeout

Instruction: `push the green square to the exit`. Seed: 220001. 100 steps, 0 pushes, 0 waits, 0 blocked moves.

![direct gate timeout](media/gate_direct_timeout.gif)

### gate — direct — distractor_contact

Instruction: `push the red square to the exit`. Seed: 220004. 5 steps, 0 pushes, 0 waits, 1 blocked moves.

![direct gate distractor_contact](media/gate_direct_distractor_contact.gif)

### gate — hybrid — success

Instruction: `push the green square to the exit`. Seed: 220001. 22 steps, 8 pushes, 1 waits, 0 blocked moves.

![hybrid gate success](media/gate_hybrid_success.gif)

### sweeper — hybrid — success

Instruction: `push the green disk to the exit`. Seed: 230003. 19 steps, 10 pushes, 0 waits, 0 blocked moves.

![hybrid sweeper success](media/sweeper_hybrid_success.gif)

### rotor — hybrid — success

Instruction: `push the amber disk to the exit`. Seed: 240001. 13 steps, 8 pushes, 0 waits, 0 blocked moves.

![hybrid rotor success](media/unseen_rotor_hybrid_success.gif)

### tower — hybrid — success

Instruction: `push the red disk to the exit`. Seed: 250003. 26 steps, 10 pushes, 0 waits, 0 blocked moves.

![hybrid tower success](media/unseen_tower_hybrid_success.gif)

### baffles — hybrid — success

Instruction: `push the blue disk to the exit`. Seed: 260004. 16 steps, 8 pushes, 0 waits, 0 blocked moves.

![hybrid baffles success](media/heldout_blue_disk_hybrid_success.gif)

### baffles — hybrid — success

Instruction: `push the red square to the exit`. Seed: 400002. 14 steps, 8 pushes, 0 waits, 0 blocked moves.

![hybrid baffles success](media/counterfactual_0.gif)

### baffles — hybrid — success

Instruction: `push the green disk to the exit`. Seed: 400002. 20 steps, 11 pushes, 0 waits, 0 blocked moves.

![hybrid baffles success](media/counterfactual_1.gif)
