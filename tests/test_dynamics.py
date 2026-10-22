"""Executable invariants, not a statistical performance test."""
import unittest
import numpy as np
from mini_vla_maze.env import World,cell,make_world,transition,hazards,advance_phase,parse_instruction
from mini_vla_maze.planner import expert,plan

class DynamicsTests(unittest.TestCase):
    def test_contact_and_wall(self):
        w=World({cell(2,5)},[cell(2,3)],[0],cell(2,2),cell(2,4),0,'open')
        w.step(1); self.assertTrue(w.success()); self.assertEqual(w.pushes,1)
        w.step(1); self.assertEqual(w.objects[0],cell(2,4)); self.assertEqual(w.collisions,1)

    def test_wait_advances_clock_and_interlock(self):
        self.assertEqual(advance_phase('gate',1,set()),2)
        self.assertEqual(advance_phase('gate',5,{cell(5,5)}),5)
        self.assertEqual(advance_phase('gate',5,set()),0)

    def test_plans_replay_without_collisions(self):
        for kind in ('open','baffles','gate','sweeper','rotor','tower'):
            found=0
            for seed in range(900,940):
                w=make_world(seed,kind); p=expert(w)
                if p is None: continue
                for a in p:
                    predicted=transition(w.walls,w.kind,set(w.objects)-{w.objects[w.target]},w.state,a)
                    w.step(a); self.assertEqual(w.state,predicted)
                    self.assertFalse(set(w.objects)&w.walls)
                    self.assertNotIn(w.pusher,w.walls)
                    self.assertFalse(hazards(w.kind,w.phase)&(set(w.objects)|{w.pusher}))
                self.assertTrue(w.success()); self.assertEqual(w.collisions,0)
                found+=1
                if found==3: break
            self.assertEqual(found,3,kind)

    def test_no_hidden_target_marker(self):
        w=make_world(12,'open'); rgb=w.render(); w.target=(w.target+1)%4
        np.testing.assert_array_equal(rgb,w.render())

    def test_instruction_roundtrip(self):
        for seed in range(20):
            w=make_world(seed)
            np.testing.assert_array_equal(parse_instruction(w.text),w.tokens)
        with self.assertRaises(ValueError): parse_instruction('move something somehow')

    def test_unsolvable_is_explicit(self):
        walls={cell(r,c) for r in range(11) for c in range(11) if r in (0,10) or c in (0,10)}
        self.assertIsNone(plan(walls,'open',set(),(cell(2,2),cell(1,1),0),cell(5,5)))

if __name__=='__main__': unittest.main()
