"""
Hand-checkable tests for ``gnn_lab.metrics``.

``metrics.py`` ships in every training loop (``evaluate_node`` calls ``macro_f1``
on every epoch), so a bug here does not give a wrong number -- it crashes every
run. These tests exist because exactly that happened once: ``confusion_matrix``
allocated a 2-D accumulator and then ``index_add_``-ed a flat index into it.
"""

from __future__ import annotations

import torch

from gnn_lab.metrics import (accuracy, confusion_matrix, macro_f1, majority_baseline,
                             per_class_report)


def test_confusion_matrix_hand_computed():
    #           true:  0  0  1  1  2  2
    #           pred:  0  1  1  1  2  0
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    pred = torch.tensor([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(pred, target, 3)
    expected = torch.tensor([[1, 1, 0],     # true 0: one right, one called 1
                             [0, 2, 0],     # true 1: both right
                             [1, 0, 1]])    # true 2: one called 0, one right
    assert cm.shape == (3, 3)
    assert torch.equal(cm, expected)
    assert int(cm.sum()) == 6               # every sample is counted exactly once


def test_confusion_matrix_diagonal_is_accuracy():
    g = torch.Generator().manual_seed(0)
    target = torch.randint(0, 7, (140,), generator=g)      # the Cora train-set shape that crashed
    pred = torch.randint(0, 7, (140,), generator=g)
    cm = confusion_matrix(pred, target, 7)
    # accuracy() averages in float32, so compare to float32 precision, not 1e-12
    assert abs(float(cm.diagonal().sum()) / 140 - accuracy(pred, target)) < 1e-6


def test_macro_f1_perfect_and_hand_computed():
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    assert abs(macro_f1(y, y, 3) - 1.0) < 1e-6

    # class 0: tp=1 fp=0 fn=1 -> P=1   R=1/2 -> F1 = 2/3
    # class 1: tp=2 fp=1 fn=0 -> P=2/3 R=1   -> F1 = 4/5
    target = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([0, 1, 1, 1])
    assert abs(macro_f1(pred, target, 2) - (2 / 3 + 4 / 5) / 2) < 1e-6


def test_macro_f1_skips_classes_absent_from_target():
    target = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([0, 0, 1, 1])
    assert abs(macro_f1(pred, target, 5) - 1.0) < 1e-6      # classes 2..4 never occur


def test_majority_baseline():
    target = torch.tensor([1, 1, 1, 0])
    assert majority_baseline(target, 2) == 0.75


def test_per_class_report_support_matches_label_counts():
    target = torch.tensor([0, 0, 0, 1, 2, 2])
    pred = torch.tensor([0, 0, 1, 1, 2, 0])
    rep = per_class_report(pred, target, 3)
    assert rep["support"] == [3, 1, 2]
