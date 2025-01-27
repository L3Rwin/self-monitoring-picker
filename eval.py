''' Evaluation of agent trajectories '''

import json
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint

import torch

pp = pprint.PrettyPrinter(indent=4)

from utils import load_datasets, load_nav_graphs


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, opts):
        self.error_margin = 3.0
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        for item in load_datasets(splits,opts):
            self.gt[item['path_id']] = item
            self.scans.append(item['scan'])
            self.instr_ids += ['%d_%d' % (item['path_id'],i) for i in range(3)]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule). '''
        gt = self.gt[int(instr_id.split('_')[0])]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        first_step= gt['path'][1]

        point_cnt=0
        point_match=0
        for idx in range(len(path)-1):
            for gt_idx in range(len(gt['path'])-1):
                if gt['path'][gt_idx]==path[idx][0]:
                    point_cnt +=1
                    if gt['path'][gt_idx+1]==path[idx+1][0]:
                    # if self.distances[gt['scan']][path[idx+1][0]][gt['path'][gt_idx+1]]<=self.error_margin:
                        point_match+=1



        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)

        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['trajectory_steps'].append(len(path)-1)
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        is_success = self.distances[gt['scan']][final_position][goal] < self.error_margin

        if self.splits == ['test']:
            self.scores['success_path_length'].append(0)
        else:
            self.scores['success_path_length'].append(
                is_success * self.distances[gt['scan']][start][goal] / max(self.distances[gt['scan']][start][goal],
                                                                           distance))

        if instr_id in ['6992_1' , '4944_0', '6684_0', '2455_2','6649_0']:
            print(instr_id,':')
            if is_success:
                print('success')
            else:
                print('not success')
                print('gt path:', gt['path'])
                print('model path:', [p[0] for p in path])

        return point_cnt, point_match

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        point_cnt=0
        point_match=0
        instr_ids = set(self.instr_ids) 
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    c1,c2=self._score_item(item['instr_id'], item['trajectory'])
                    point_cnt +=c1
                    point_match+=c2
        assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                       % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
        assert len(self.scores['nav_errors']) == len(self.instr_ids)
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths']),
            'spl': np.average(self.scores['success_path_length'])
        }
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))
        print("acc for single step:", point_match/point_cnt)
        return score_summary, self.scores
