import math
from clean_ppo_reward import transition_reward


def test_left_right_symmetry():
    for h in (0.0, 0.4, 1.2, 2.4):
        a=transition_reward(.6,.58,h,.17)
        b=transition_reward(.6,.58,-h,.17)
        assert abs(a-b)<1e-12


def test_goal_side_pass_is_discouraged():
    straight=transition_reward(.50,.49,0.1,.18)
    side_fast=transition_reward(.50,.49,math.pi/2,.18)
    side_slow=transition_reward(.50,.49,math.pi/2,.03)
    assert straight>side_slow>side_fast


def test_regression_loses_to_progress():
    toward=transition_reward(.50,.47,math.pi/2,.10)
    away=transition_reward(.47,.50,math.pi/2,.10)
    assert toward>away and away<0


def test_terminal_ordering():
    assert transition_reward(.3,.19,0,0,success=True)>0
    assert transition_reward(.3,.25,0,0,collision=True)<transition_reward(.3,.25,0,0,timed_out=True)


if __name__=="__main__":
    test_left_right_symmetry();test_goal_side_pass_is_discouraged()
    test_regression_loses_to_progress();test_terminal_ordering()
    print("CLEAN_PPO_REWARD_TEST_PASS")
