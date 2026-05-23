import csv
import json
from pathlib import Path
import torch


@torch.no_grad()
def params_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().float()


class Metric:
    name = "metric"

    def update(self, model, row):
        raise NotImplementedError


class OscillationFraction(Metric):
    def __init__(self, name="oscillation_fraction", eps=0.0):
        self.name = name
        self.eps = eps
        self.prev = None
        self.last = None

    def update(self, model, row):
        current = params_vector(model)
        value = None
        if self.prev is not None and self.last is not None:
            old_step = self.last - self.prev
            new_step = current - self.last
            active = (old_step.abs() > self.eps) & (new_step.abs() > self.eps)
            changed = active & (old_step * new_step < 0)
            value = float(changed.float().mean().item())
        self.prev, self.last = self.last, current
        return value


class TrainingLogger:
    def __init__(self, metrics=None):
        self.metrics = metrics or []
        self.rows = []

    def prime(self, model):
        for metric in self.metrics:
            metric.update(model, {})

    def log(self, model, **fields):
        row = dict(fields)
        for metric in self.metrics:
            row[metric.name] = metric.update(model, row)
        self.rows.append(row)
        return row

    def save_jsonl(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")

    def save_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for row in self.rows for k in row})
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)
