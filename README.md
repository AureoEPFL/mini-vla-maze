# Mini-VLA Maze

Can a small vision-and-language policy push the requested object through a maze without hitting the others? This project tests that question in a 2D simulator with static walls, moving barriers and timed gates.

![A learned visual map guides A* through three timed passages](media/viewer_airlocks.gif)

*The demo uses the network to read the scene, then A* to plan around the timed gates. The network itself does not generate this route.*

## What the results show

![Benchmark scenes and episode success](media/benchmark_illustrated.png)

Each panel shows one maze and the fraction of 16 solvable scenes completed. Blue and purple are direct action predictions; green is learned perception followed by A*; grey is the simulator's oracle planner. The error bars are 95% Wilson intervals. The direct policy succeeds on 0–5 scenes per layout, while the hybrid completes all 16 in these small test sets. Layouts marked `*` were held out during training.

![Training losses and validation accuracy](media/learning.png)

The model learns target and object locations almost perfectly on validation frames, but next-action accuracy levels off at 86.9%. That gap helps explain why good visual grounding does not guarantee a successful multi-step push. More results, failure cases and limitations are in [RESULTS.md](RESULTS.md).

## How it works

A compact network reads a rendered image and a short instruction. It predicts the target cell, a semantic grid and a short action chunk. We compare its direct actions with a hybrid controller that runs A* over the network's predicted grid using the simulator's known movement rules. The hybrid's maze-solving ability comes from A*, not from a learned action policy.

![Model architecture](media/paper_architecture.png)

The model has 401,729 parameters and is trained on synthetic expert demonstrations. This is a small, controlled simulation: it does not model robot hardware, force control or real-world camera variation.

## Run it

Python 3.10+; CPU is enough. From this folder:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m mini_vla_maze.demo --kind gate --controller hybrid
```

The trained checkpoint and recorded results are included. Open [`gallery.html`](gallery.html) locally to browse the saved episodes, or start with [notebook 02](notebooks/02_maze_lab.ipynb). [Reproduction steps](docs/REPRODUCE.md) include retraining and Colab validation.

The detailed [methods report](docs/METHODS.md) describes the data, architecture and evaluation protocol. [Environment guide](docs/ENVIRONMENTS.md) explains the obstacles; [applications](docs/APPLICATIONS.md) discusses what would need to change for a real robot.
