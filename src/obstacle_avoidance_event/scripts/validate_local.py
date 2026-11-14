#!/usr/bin/env python
"""Single offline validation entry. Never starts ROS or vehicle transports."""
import os,sys,unittest,json,argparse
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from closed_loop_demo import run

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.discover(os.path.join(ROOT,'tests'))
    checks=unittest.TextTestRunner(verbosity=1).run(suite)
    cases={}
    for name,kind,obstacles,delay in [
        ('straight','straight',[],0),('s_curve','s_curve',[],0),
        ('single','straight',[(5,0,.2)],0),
        ('alternating','straight',[(4,.2,.2),(7,-.2,.2)],0),
        ('alternating_delayed','straight',[(4,.2,.2),(7,-.2,.2)],3)]:
        result=run(kind,duration=95,obstacles=obstacles,delay_steps=delay,goal_x=10 if obstacles else 8)
        cases[name]=result
    passed=checks.wasSuccessful() and all(r['goal_reached'] and r['reason'] is None and r['motion_safe'] and r['stopped'] for r in cases.values())
    directory=os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(directory):os.makedirs(directory)
    with open(args.output,'w') as f:
        json.dump(dict(passed=passed,test_count=checks.testsRun,vehicle_output=False,cases=cases),f,indent=2)
    print('local validation:',passed,'tests:',checks.testsRun)
    return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
