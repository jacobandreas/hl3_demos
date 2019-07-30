#!/usr/bin/env python2

import itertools as it
import json
import numpy as np
import os
import pickle
from tqdm import tqdm

DATA_DIR = "../data"
SEG_DIR = "instance_segmentation_data"
HOUSE_DIR = "houses"
SNAPSHOT_TEMPLATE = "snapshots/{house}-{plan_type}-{step}.npz"

ACTION_STOP = 0
ACTION_GO = 1
TASK_PAD = "*pad*"
TASK_SOS = "*sos*"
TASK_EOS = "*eos*"

GOOD_PARTS = {"wall", "roof"}
SCHEMATIC_SIZE = (20, 20, 20)

def build_schematic(full_schematic, segmentation, annotations):
    keep_schematic = np.zeros(full_schematic.shape, dtype=np.int32)
    for i, annotation in enumerate(annotations):
        if annotation not in GOOD_PARTS:
            continue
        keep_schematic[segmentation == i] = full_schematic[segmentation == i]
    if (keep_schematic > 0).sum() == 0:
        return None
    return keep_schematic

def build_plan_flat(schematic, annotation, annotation_vocab):
    if annotation not in annotation_vocab:
        annotation_vocab[annotation] = len(annotation_vocab)
    plan = []
    snapshots = []
    schematic_so_far = np.zeros(SCHEMATIC_SIZE)
    for x, y, z in it.product(*[range(d) for d in schematic.shape]):
        if schematic[x, y, z] == 0:
            continue
        plan.append({
            "annotation": annotation_vocab[annotation], 
            "action": ACTION_GO, 
            "argument": (x, y, z, schematic[x, y, z]), 
        })
        snapshots.append(schematic_so_far.copy())
        if x < SCHEMATIC_SIZE[0] and y < SCHEMATIC_SIZE[1] and z < SCHEMATIC_SIZE[2]:
            schematic_so_far[x, y, z] = schematic[x, y, z]
    plan.append({
        "annotation": annotation_vocab[annotation],
        "action": ACTION_STOP,
        "argument": None
    })
    snapshots.append(schematic_so_far.copy())
    return plan, snapshots

def build_plan_hier(schematic, segmentation, annotations, annotation_vocab):
    plan = []
    snapshots = []
    for i, annotation in enumerate(annotations):
        if annotation not in GOOD_PARTS:
            continue
        part_schematic = np.zeros(schematic.shape, dtype=np.int32)
        part_schematic[segmentation == i] = schematic[segmentation == i]
        part_plan_hier, part_snapshots_hier = build_plan_flat(part_schematic, annotation, annotation_vocab)
        plan += part_plan_hier
        snapshots += part_snapshots_hier
    return plan, snapshots

def generate_dataset():
    annotation_vocab = {TASK_PAD: 0, TASK_SOS: 1, TASK_EOS: 2}
    plans_flat = []
    plans_hier = []
    with open(os.path.join(DATA_DIR, SEG_DIR, "training_data.pkl"), "rb") as seg_f:
        segments = pickle.load(seg_f)
    for full_schematic, segmentation, annotations, house_id in tqdm(segments):
        schematic = build_schematic(full_schematic, segmentation, annotations)
        if schematic is None:
            continue
        plan_flat, snapshots_flat = build_plan_flat(schematic, "house", annotation_vocab)
        for i in range(len(plan_flat)):
            snapshot_name = SNAPSHOT_TEMPLATE.format(house=house_id, plan_type="flat", step=i)
            np.savez(snapshot_name, data=snapshots_flat[i])
            plan_flat[i]["snapshot"] = snapshot_name
        plans_flat.append(plan_flat)

        plan_hier, snapshots_hier = build_plan_hier(schematic, segmentation, annotations, annotation_vocab)
        plans_hier.append(plan_hier)
        for i in range(len(plan_hier)):
            snapshot_name = SNAPSHOT_TEMPLATE.format(house=house_id, plan_type="hier", step=i)
            np.savez(snapshot_name, data=snapshots_hier[i])
            plan_hier[i]["snapshot"] = snapshot_name
        plans_hier.append(plan_hier)

if __name__ == "__main__":
    generate_dataset()
