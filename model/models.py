from constants import TASK_PAD

import numpy as np
import torch
from torch import nn, optim

N_EMB_TASK = 64
N_EMB_ACTION = 64
N_HIDDEN = 1024

class RnnModelState(object):
    def __init__(self):
        pass

class RnnModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        
        self.annotation_emb = nn.Embedding(dataset.n_annotations, N_EMB_TASK)
        self.annotation_rnn = nn.LSTM(N_EMB_TASK, N_HIDDEN)
        self.annotation_out = nn.Linear(N_HIDDEN, dataset.n_annotations)
        self.task_loss = nn.CrossEntropyLoss(ignore_index=dataset.annotation_vocab[TASK_PAD])

        self.action_emb = nn.Embedding(3, N_EMB_ACTION)
        self.pos_emb = nn.Embedding(1000, N_EMB_ACTION) # TODO
        self.mat_emb = nn.Embedding(1000, N_EMB_ACTION) # TODO
        self.action_features = nn.Linear(N_EMB_TASK + 5 * N_EMB_ACTION, N_HIDDEN)
        self.action_rnn = nn.LSTM(N_HIDDEN, N_HIDDEN)
        self.action_out = nn.Linear(N_HIDDEN, 3)
        self.x_out = nn.Linear(N_HIDDEN, 1000)
        self.y_out = nn.Linear(N_HIDDEN, 1000)
        self.z_out = nn.Linear(N_HIDDEN, 1000)
        self.mat_out = nn.Linear(N_HIDDEN, 1000)
        self.action_loss = nn.CrossEntropyLoss(ignore_index=2)
        self.arg_loss = nn.CrossEntropyLoss(ignore_index=999)
        
    def score_task(self, annotations, snapshots):
        ctx = annotations[:-1, :]
        tgt = annotations[1:, :]
        tgt = tgt.view(-1)
        annotation_emb = self.annotation_emb(ctx)
        annotation_hid, _ = self.annotation_rnn(annotation_emb)
        annotation_out = self.annotation_out(annotation_hid).view(tgt.shape[0], -1)
        loss = self.task_loss(annotation_out, tgt)
        return loss

    def score_actions(self, annotations, actions, xs, ys, zs, mats):
        c_annotations = annotations[:-1, :]
        c_actions = actions[:-1, :]
        c_xs = xs[:-1, :]
        c_ys = ys[:-1, :]
        c_zs = zs[:-1, :]
        c_mats = mats[:-1, :]

        t_actions = actions[1:, :].view(-1)
        t_xs = xs[1:, :].view(-1)
        t_ys = ys[1:, :].view(-1)
        t_zs = zs[1:, :].view(-1)
        t_mats = mats[1:, :].view(-1)

        annotation_emb = self.annotation_emb(c_annotations)
        action_emb = self.action_emb(c_actions)
        x_emb = self.pos_emb(c_xs)
        y_emb = self.pos_emb(c_ys)
        z_emb = self.pos_emb(c_zs)
        mat_emb = self.mat_emb(c_mats)
        action_inputs = torch.cat([annotation_emb, action_emb, x_emb, y_emb, z_emb, mat_emb], dim=2)
        action_features = self.action_features(action_inputs)
        action_hid, _ = self.action_rnn(action_features)

        action_out = self.action_out(action_hid).view(t_actions.shape[0], -1)
        x_out = self.x_out(action_hid).view(t_xs.shape[0], -1)
        y_out = self.y_out(action_hid).view(t_ys.shape[0], -1)
        z_out = self.z_out(action_hid).view(t_zs.shape[0], -1)
        mat_out = self.mat_out(action_hid).view(t_mats.shape[0], -1)

        loss = (
            self.action_loss(action_out, t_actions)
            + self.action_loss(x_out, t_xs)
            + self.action_loss(y_out, t_ys)
            + self.action_loss(z_out, t_zs)
            + self.action_loss(mat_out, t_mats)
        )
        return loss
