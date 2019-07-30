#!/usr/bin/env python3

from constants import ACTION_STOP, TASK_SOS, TASK_EOS
from models import RnnModel

import json
import pickle
import numpy as np
import os
import torch
from torch import nn, optim

DEVICE = torch.device("cuda:0")

def _to_tensor(tensors):
    assert isinstance(tensors, tuple)
    return tuple(torch.tensor(t).to(DEVICE) for t in tensors)

class Dataset(object):
    def __init__(self, annotations, actions, arguments, annotation_vocab, snapshot_dir):
        self.annotations = annotations
        self.actions = actions
        self.arguments = arguments
        self.annotation_vocab = annotation_vocab
        self.snapshot_dir = snapshot_dir
        
        self.n_annotations = len(annotation_vocab)

    def load_snapshots(self, indices):
        #return [[] for _ in indices]
        snapshots = []
        for index in indices:
            data = np.load(os.path.join(self.snapshot_dir, "{}.npz".format(index)))["data"]
            snapshots.append(data)
        return snapshots
        
    def sample_task_batch(self, batch_size):
        indices = np.random.randint(len(self.annotations), size=batch_size)
        all_snapshots = self.load_snapshots(indices)
        all_actions = [self.actions[i] for i in indices]
        all_annotations = [self.annotations[i] for i in indices]
        batch_annotations = []
        batch_snapshots = []
        for a_snapshots, actions, a_annotations in zip(all_snapshots, all_actions, all_annotations):
            boundaries = [j for j in range(len(actions)) if actions[j] == ACTION_STOP]
            annotations = [a_annotations[j-1] for j in boundaries]
            annotations = [self.annotation_vocab[TASK_SOS]] + annotations + [self.annotation_vocab[TASK_EOS]]
            batch_annotations.append(annotations)
            snapshots = [a_snapshots[j-1] for j in boundaries]
            batch_snapshots.append(snapshots)
        max_len = max(len(s) for s in batch_annotations)
        annotation_data = np.zeros((max_len, batch_size), dtype=np.int64)
        snapshot_data = np.zeros((max_len, batch_size) + batch_snapshots[0][0].shape, dtype=np.int64)
        #snapshot_data = np.zeros(1)
        for i in range(batch_size):
            annotation_data[:len(batch_annotations[i]), i] = batch_annotations[i]
            snapshot_data[:len(batch_snapshots[i]), i, ...] = batch_snapshots[i]
        return (annotation_data, snapshot_data)

    def sample_action_batch(self, batch_size, max_len):
        indices = np.random.randint(len(self.annotations), size=batch_size)
        batch_annotations = []
        batch_actions = []
        batch_xs = []
        batch_ys = []
        batch_zs = []
        batch_mats = []
        for i in indices:
            offset = np.random.randint(max(1, len(self.annotations[i])-max_len))
            annotations = self.annotations[i][offset:offset+max_len]
            actions = self.actions[i][offset:offset+max_len]
            arguments = self.arguments[i][offset:offset+max_len]
            xs, ys, zs, mats = zip(*arguments)
            batch_annotations.append(annotations)
            batch_actions.append(actions)
            batch_xs.append(xs)
            batch_ys.append(ys)
            batch_zs.append(zs)
            batch_mats.append(mats)
        annotation_data = np.zeros((batch_size, max_len), dtype=np.int64)
        action_data = np.ones((batch_size, max_len), dtype=np.int64) * 2
        x_data = np.ones((batch_size, max_len), dtype=np.int64) * 0
        y_data = np.ones((batch_size, max_len), dtype=np.int64) * 0
        z_data = np.ones((batch_size, max_len), dtype=np.int64) * 0
        mat_data = np.ones((batch_size, max_len), dtype=np.int64) * 0
        for i in range(len(batch_annotations)):
            annotation_data[i, :len(batch_annotations[i])] = batch_annotations[i]
            action_data[i, :len(batch_actions[i])] = batch_actions[i]
            x_data[i, :len(batch_xs[i])] = batch_xs[i]
            y_data[i, :len(batch_ys[i])] = batch_ys[i]
            z_data[i, :len(batch_zs[i])] = batch_zs[i]
            mat_data[i, :len(batch_mats[i])] = batch_mats[i]
        return (annotation_data, action_data, x_data, y_data, z_data, mat_data)
        
def make_dataset(data_type):
    path_pkl_data = "../dataset/{}.pkl".format(data_type)
    ann_vocab_json_data = "../dataset/annotation_vocab.json"
    snapshot_dir = "../dataset/snapshots/{}".format(data_type)
    with open(path_pkl_data, "rb") as f:
        paths = pickle.load(f)
    with open(ann_vocab_json_data) as f:
        annotation_vocab = json.load(f)
    annotations = []
    actions = []
    arguments = []
    for path in paths:
        p_annotations, p_actions, p_arguments = zip(*path)    
        annotations.append(p_annotations)
        actions.append(p_actions)
        arguments.append(p_arguments)
    return Dataset(annotations, actions, arguments, annotation_vocab, snapshot_dir)

INIT_LR = 0.001
BATCH_SIZE = 20
MAX_LEN = 100
N_ITERS = 2000
LR_STEP = N_ITERS / 2
N_LOG = 100

def train(data_type):
    print(data_type)
    assert data_type in ("flat", "hier")
    dataset = make_dataset(data_type)
    model = RnnModel(dataset).to(DEVICE)
    train_tasks(dataset, model)
    #train_actions(dataset, model)
    
def train_tasks(dataset, model):
    print("train_tasks")
    opt = optim.Adam(model.parameters(), INIT_LR)
    opt_sched = optim.lr_scheduler.StepLR(opt, LR_STEP)
    for i in range(N_ITERS):
        batch = _to_tensor(dataset.sample_task_batch(BATCH_SIZE))
        opt.zero_grad()
        loss = model.score_task(*batch)
        loss.backward()
        opt.step()
        opt_sched.step()
        if (i+1) % N_LOG == 0:
            print("{:0.3f}".format(loss.item()))
        
def train_actions(dataset, model):
    print("train_actions")
    opt = optim.Adam(model.parameters(), INIT_LR)
    opt_sched = optim.lr_scheduler.StepLR(opt, LR_STEP)
    for i in range(N_ITERS):
        batch = _to_tensor(dataset.sample_action_batch(BATCH_SIZE, MAX_LEN))
        opt.zero_grad()
        loss = model.score_actions(*batch)
        loss.backward()
        opt.step()
        opt_sched.step()
        if (i+1) % N_LOG == 0:
            print("{:0.3f}".format(loss.item()))

if __name__ == "__main__":
    train("hier")
    train("flat")
