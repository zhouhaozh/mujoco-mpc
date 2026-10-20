"""录像截断约束；不需要渲染器或仿真运行库。"""
import unittest
from record_prediction_demo import trim_second_contact,prediction_until_second_contact
class ClippingTest(unittest.TestCase):
 def test_exact_contact_count_wins_over_height_noise(self):
  rows=[dict(t=i,ground_contacts=n,ball=[0,0,z]) for i,(n,z) in enumerate([(0,.04),(0,.03),(0,.04),(1,.033),(1,.4),(2,.033),(2,.2)])]
  self.assertEqual(trim_second_contact(rows),rows[:6])
 def test_single_bounce_keeps_remaining_flight(self):
  rows=[dict(ground_contacts=n,ball=[0,0,.1]) for n in [0,1,1]]
  self.assertEqual(trim_second_contact(rows),rows)
 def test_prediction_does_not_rejoin_after_second_ground(self):
  grid=[[0,0,0,.4,-1,False],[.1,1,0,.033,1,True],[.3,2,0,.2,-1,True],[.5,3,0,.032,-2,True],[.6,4,0,.2,1,True]]
  pts=prediction_until_second_contact(grid,0,0)
  self.assertEqual(pts[-1],[3,0,.033]);self.assertEqual(len(pts),4)
 def test_postbounce_cache_stops_at_next_ground(self):
  grid=[[1,0,0,.4,-1,True],[1.1,1,0,.032,-2,True],[1.2,2,0,.2,1,True]]
  self.assertEqual(prediction_until_second_contact(grid,1,1),[[0,0,.4],[1,0,.033]])
 def test_old_trace_stops_at_second_height_minimum(self):
  rows=[dict(ball=[0,0,z]) for z in [.5,.04,.3,.6,.04,.3,.6]]
  self.assertEqual(len(trim_second_contact(rows)),5)
if __name__=='__main__':unittest.main()
