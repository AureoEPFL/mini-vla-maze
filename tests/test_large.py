import unittest
from pathlib import Path
import numpy as np
import torch
from mini_vla_maze.env import make_world
from mini_vla_maze.large import make_arena,search,step_state,render_labels,perceive,door_cells
from mini_vla_maze.evaluate import load_model

class LargeTests(unittest.TestCase):
    def test_same_training_tiles(self):
        for seed in range(5):
            w=make_world(seed,'baffles')
            np.testing.assert_array_equal(render_labels(w.labels()),w.render())
            w=make_world(seed,'gate')
            # The original hazard stroke can spill one pixel beyond its tile.
            # The large renderer clips that spill through its raster order.
            # Interior pixels retain the learned colour/shape vocabulary.
            actual,expected=render_labels(w.labels()),w.render()
            for r in range(11):
                for c in range(11):
                    box=np.s_[r*8+1:r*8+7,c*8+1:c*8+7]
                    np.testing.assert_array_equal(actual[box],expected[box])

    def test_long_plan_and_collision_replay(self):
        for kind,seed in [('switchback',800000),('catacombs',810001),('airlocks',820002)]:
            w=make_arena(seed,kind);fixed=set(w.objects)-{w.objects[w.target]}
            path,_=search(w.n,w.walls,w.doors,fixed,w.state,w.goal)
            self.assertIsNotNone(path);self.assertGreater(len(path),40)
            for a in path:
                expected=step_state(w.n,w.walls,w.doors,fixed,w.state,a);w.step(a)
                self.assertEqual(expected,w.state)
                self.assertNotIn(w.pusher,w.walls)
                self.assertFalse(set(w.objects)&w.walls)
                self.assertFalse(door_cells(w.doors,w.phase)&(set(w.objects)|{w.pusher}))
            self.assertTrue(w.success());self.assertEqual(w.collisions,0)
            if kind=='airlocks':self.assertGreater(w.waits,0)

    def test_visual_heads_scale_without_action_head(self):
        root=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
        model=load_model(root/'results/maze/weights.pt')
        w=make_arena(800000);labels,attention=perceive(model,w.render(),w.tokens)
        self.assertEqual(labels.shape,(25,25));np.testing.assert_array_equal(labels,w.labels())
        self.assertEqual(attention.argmax(),w.objects[w.target])

if __name__=='__main__':unittest.main()
