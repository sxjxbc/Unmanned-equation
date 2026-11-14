#!/usr/bin/env python3
"""Offline ablations only: do not install these parameters on a vehicle."""
import argparse
import csv
import json
import math
from pathlib import Path
from legacy_loop_validation import run_case


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='exports/loop_response_20261019')
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    experiments=[('unchanged',{}),
                 ('no_error_filter',{'~error_filter_alpha':1.}),
                 ('lighter_error_filter',{'~error_filter_alpha':.4}),
                 ('double_command_slew',{'~max_steering_step':.024}),
                 ('global_target_only',{'~local_blend_two_pairs':0.,'~local_blend_one_pair':0.}),
                 ('lighter_filter_double_slew',{'~error_filter_alpha':.4,'~max_steering_step':.024})]
    results=[]
    for gain in (.25,.5,1.):
        for name,overrides in experiments:
            summary,records=run_case(name,initial_yaw=math.radians(10),steering_gain=gain,
                                     parameter_overrides=overrides)
            results.append(summary)
            with (output/('%s_gain_%s.csv'%(name,gain))).open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]))
                writer.writeheader();writer.writerows(records)
            print('%s gain=%.2f max_y=%.3f final_y=%.3f rate_limited=%.1f%% stop=%s' %
                  (name,gain,summary['max_abs_y'],summary['final_y'],
                   100*summary['rate_limited_fraction'],summary['stop_reason']))
    report=dict(production_parameters_changed=False,vehicle_calibration_known=False,
                warning='Hypothesis tests, not calibrated tuning recommendations. No ROS/CAN.',
                fixed_assumptions=dict(wheelbase=1.6,speed=2.,actuator_lag=.1),experiments=results)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
