# Obstacles and tasks

## Implemented

| Mechanism | Challenge | Exact rule |
|---|---|---|
| Baffles | Turn, reposition behind the object, avoid dead ends | Two offset wall segments; one segment length varies with seed |
| Timed door | Approach and wait for an opening | Single-cell passage, closed for phases 0-2 and open for 3-5 |
| Sweeper | Coordinate passage with a moving blocker | One blocked cell travels along a six-phase horizontal sequence |
| Raster turnstile | Route around alternating arms | Four axial cells alternate with four diagonal cells; centre stays fixed |
| Tower | Navigate a narrowing silhouette and interior obstacles | Fixed stylised tower geometry, unseen in training |

All mechanisms pause if their next pose would overlap an object or the pusher. This prevents crushing/teleportation and is part of the published task definition. The turnstile is a discrete approximation, not a continuous revolving door. Some layouts allow detours, so successful passage does not necessarily demonstrate learning to wait; the logged wait count makes this inspectable.

## Additional experiments worth implementing separately

1. **Pressure-plate door:** leave one object on a switch while moving another. Requires multi-object planning and changes the current protected-distractor rule.
2. **One-way flap:** crossing is allowed in only one direction; creates irreversible decisions.
3. **Conveyor:** external motion displaces the object after the robot acts. Requires predicting environmental dynamics.
4. **Spring gate:** opens only while pushed and closes after release. Adds contact and timing dependencies.
5. **Partial occlusion:** a wall blocks the camera view. Requires memory and belief-state estimation, not just a larger image encoder.
6. **Unknown obstacle period:** infer motion from consecutive frames instead of supplying a phase sensor and known schedule.

These extensions are proposals, not included capabilities. A continuously rotating four-panel door would belong in a rigid-body simulator with swept collision checking; a visually smooth animation alone would not implement those mechanics.
