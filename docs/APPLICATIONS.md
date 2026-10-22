# Potential applications and transfer requirements

The current checkpoint solves a synthetic benchmark. The following are research directions, not validated deployments.

| Direction | Transferable idea | What would have to change |
|---|---|---|
| Tabletop sorting | Select an object from a command, plan a route around obstacles | Real image/action demonstrations, calibrated camera, gripper/pusher dynamics and continuous control |
| Warehouse transport | Language-selected destination and time-dependent crossings | Robot localisation, occupancy estimation, vehicle geometry and uncertain moving obstacles |
| Laboratory sample handling | Attribute-based selection and protected neighbouring objects | Instance identity, depth, force limits and manipulation data from the actual equipment |
| Surgical camera research | Instruction-conditioned visual attention and feedback | Endoscopic data, articulated camera kinematics, task-specific simulation and an appropriate control interface |

For waste sorting, a labelled image dataset could help train perception, but cannot alone supply pushing or grasping actions. For robotic control, one needs synchronised observations and robot actions, or a simulator that can generate those actions. The semantic attributes and grammar would also need to reflect the task.

The most direct next step is tabletop pushing in a continuous 2D or 3D simulator: retain language-conditioned selection, replace the grid transition model, train a low-level controller, and measure closed-loop success on new object placements and friction settings. This would test whether the benchmark's useful abstractions survive more realistic contact.
