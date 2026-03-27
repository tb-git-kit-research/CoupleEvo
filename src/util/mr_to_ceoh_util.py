import os
import json

import numpy as np
from util.paths_util import get_working_dir, read_all_json_files

from problems.multibay_reshuffeling.bay.virtual_lane import VirtualLane


def convert_list_to_vl(vl_list: list, ap_ids: list):
    virtual_lanes = []
    for i in range(len(vl_list)):
        new_lane = VirtualLane()
        new_lane.ap_id = ap_ids[i]
        new_lane.stacks = np.array(vl_list[i])
        virtual_lanes.append(new_lane)
    return virtual_lanes


def convert_vl_to_list(virtual_lanes):
    return [list(v.stacks) for v in virtual_lanes]


def get_ap_ids_from_vl(virtual_lanes: list):
    return [vl.ap_id for vl in virtual_lanes]


def get_virtual_lane_score(virtual_lanes: list, reversed=False):
    return sum([1 for lane in virtual_lanes if sorted(lane, reverse=reversed) != lane])


def get_access_directions(config):
    access_directions = config['bay_info']['0']['access_directions']
    return {
        "north": 'north' in access_directions,
        "east": 'east' in access_directions,
        "south": 'south' in access_directions,
        "west": 'west' in access_directions
    }

from typing import List, Union


def _parse_range(token: str) -> List[Union[int, float]]:
    """Parses a numeric token that may be a single value or a range (e.g., 4-5)."""
    if '-' in token:
        start, end = token.split('-')
        # Handle integer or float
        if '.' in start or '.' in end:
            return [float(start), float(end)]
        return [int(start), int(end)]
    else:
        return [float(token)] if '.' in token else [int(token)]


def read_config_from_filename(config: str):
    # Example: exp_bay_4-5_wh_1-2_fill_0.6_seed_0-9_robots_2_ad_1010.json
    parts = config.replace(".json", "").split("_")

    data = {}

    # iterate through tokens in pairs: key, value
    i = 1  # skip initial 'exp' or prefix
    while i < len(parts):
        key = parts[i]

        # ensure we have a value
        if i + 1 >= len(parts):
            break
        val = parts[i + 1]

        if key == "bay":
            data["bay"] = _parse_range(val)

        elif key in ("wh", "warehouse"):
            data["warehouse"] = _parse_range(val)

        elif key == "fill":
            data["fill"] = _parse_range(val)

        elif key == "seed":
            parsed = _parse_range(val)
            # seeds usually represent a full integer span
            if len(parsed) == 2 and isinstance(parsed[0], int):
                data["seed"] = list(range(parsed[0], parsed[1] + 1))
            else:
                data["seed"] = parsed

        elif key == "robots":
            data["robots"] = int(val)

        elif key in ("ad", "access_directions"):
            data["access_directions"] = val

        elif key == "priority":
            data["priority"] = _parse_range(val)

        i += 2
    print("EVAL CONFIG EXTRACTED FROM FILENAME")
    print(data)
    return data


def load_experiments(file):
    if os.getenv('BASE_PATH') is None:
        base_path = get_working_dir()
    else:
        base_path = os.getenv('BASE_PATH')

    instance_path = os.getenv('INSTANCES_PATH')
    experiment_path = os.path.join(base_path, "data", "eoh_experiment_config")

    experiment_instances = []
    experiment_solution = []

    try:
        f_exp = open(os.path.join(experiment_path, file))
        data = json.load(f_exp)
    except:
        try:
            data = read_config_from_filename(file)
        except:
            print("[ERROR]: load_experiments failed - experiment file does not exists!")
            print(f"instance_path {instance_path}")
            print(f"experiment_path {experiment_path}")
            exit(1)

    for s in data["seed"]:
        for bay in data["bay"]:
            for warehouse in data["warehouse"]:
                for fill in data["fill"]:
                    for priority in data["priority"]:

                        try:
                            f_inst = open(os.path.join(instance_path,
                                                       f'test_file_Size_{bay}x{bay}_'
                                                       f'Layout_{warehouse}x{warehouse}_'
                                                       f'fill_lvl_{fill}_'
                                                       f'seed_{s}_'
                                                       f'max_p_{priority}_'
                                                       f'ad_access_directions_'
                                                       f'{data["access_directions"]}.json'))

                            instance = json.load(f_inst)
                            experiment_instances.append(instance)
                            experiment_solution.append(instance['h_initial'])

                        except:
                            print("[ERROR]: load_experiments failed - file does not exists!")
                            print(f"[ERROR]: seed: {s}, bay: {bay}, warehouse: {warehouse}, "
                                  f"fill: {fill}, priority: {priority}")
                            print("---> continue with other files...")

    return experiment_instances, experiment_solution


def generate_instances(number_of_instances: int, path=get_working_dir()) -> list:
    path = os.getenv('INSTANCES_PATH')
    all_files = read_all_json_files(path)

    if number_of_instances > len(all_files):
        raise Exception('Not enough experiment instances')

    instance_configs = []

    for i in range(number_of_instances):
        f = open(os.path.join(path, all_files[i]))
        data = json.load(f)
        instance_configs.append(data)

    return instance_configs


def create_virtual_lane(data):
    lanes = []
    for virtual_lane in data["virtual_lanes"]:
        new_lane = VirtualLane()
        new_lane.ap_id = virtual_lane["ap_id"]
        new_lane.stacks = np.array(virtual_lane["stacks"])
        lanes.append(new_lane)

    return lanes


def create_virtual_lane_reversed(data, max_value):
    lanes = []
    for virtual_lane in data["virtual_lanes"]:
        new_lane = VirtualLane()
        new_lane.ap_id = virtual_lane["ap_id"]

        # Reverse for LMM understanding
        new_lane.stacks = np.array(virtual_lane["stacks"])

        print(new_lane.stacks)

        new_lane.stacks = new_lane.stacks[::-1]

        # Flip the priorities
        new_lane.stacks = np.array([min(1, x) * abs(x - (max_value + 1)) for x in new_lane.stacks])

        print(new_lane.stacks)
        print()

        lanes.append(new_lane)

    return lanes


def create_lanes(wh, reversed=False):
    lanes = wh.virtual_lanes

    loads = []
    slots = []
    for i in range(len(lanes)):
        if np.any(lanes[i].stacks != 0):
            loads.append(i)
        if 0 in lanes[i].stacks:
            slots.append(i)

    successors = []
    moves = []
    from_lanes = loads
    to_lanes = slots
    for from_lane in from_lanes:
        for to_lane in to_lanes:

            if from_lane == to_lane:
                continue

            new_lane = lanes[:]
            if reversed:
                new_lane[from_lane], stacks = lanes[from_lane].remove_load_reversed()
                new_lane[to_lane] = lanes[to_lane].add_load_reversed(stacks)
            else:
                new_lane[from_lane], stacks = lanes[from_lane].remove_load()
                new_lane[to_lane] = lanes[to_lane].add_load(stacks)

            successors.append(new_lane)
            moves.append([lanes[from_lane].ap_id, lanes[to_lane].ap_id])

    return successors, moves
