import pickle
from typing import List, Optional
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

import wandb
import wandb.sklearn

from .store import ActivationStore


class ProbeTrainer:
    def __init__(
        self,
        store: ActivationStore,
        label_names: List[str],
        wandb_project: Optional[str] = None,
        model_class=LogisticRegression,
        **model_kwargs,
    ):
        self.store = store
        self.probes = {}
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.project_name = wandb_project
        self.label_names = label_names
        if self.project_name is not None:
            wandb.init(project=self.project_name)

    def prepare_data(self, hook_name: str):
        X = np.concatenate(self.store.activations[hook_name], axis=0)
        y = np.array(self.store.labels)
        B, T, D = X.shape
        X = X.reshape(B * T, D)  # (BS * T, D)
        y = np.repeat(y, T)  # (BS * T)

        if len(X) == 0 or len(y) == 0:
            raise ValueError(f"No activations found for hook: {hook_name}")
        if len(X) != len(y):
            raise ValueError(
                f"Length of activations and labels must be the same. Got {len(X)} and {len(y)}"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )

        uniq_train_cls, uniq_test_cls = np.unique(y_train), np.unique(y_test)
        if len(uniq_train_cls) < 2:
            raise ValueError(f"Train data contains only one class: {uniq_train_cls[0]}")
        if len(uniq_test_cls) < 2:
            raise ValueError(f"Test data contains only one class: {uniq_test_cls[0]}")
        return X_train, X_test, y_train, y_test

    def train_probe(self, hook_name: str):
        X_train, X_test, y_train, y_test = self.prepare_data(hook_name)
        clf = self.model_class(**self.model_kwargs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_probas = clf.predict_proba(X_test)
        print(f"Probe Metrics for {hook_name}:")
        print(classification_report(y_test, y_pred, target_names=self.label_names))

        if self.project_name is not None:
            try:
                wandb.sklearn.plot_classifier(
                    clf,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    y_pred,
                    y_probas,
                    self.label_names,
                    feature_names=None,
                    model_name=hook_name,
                    log_learning_curve=True,
                )
            except ValueError as e:  # calibration_curve can fail
                wandb.termwarn(f"Could not plot classifier for {hook_name}: {e}")

                wandb.sklearn.plot_roc(y_test, y_probas, self.label_names)
                wandb.termlog("Logged roc curve.")

                wandb.sklearn.plot_precision_recall(y_test, y_probas, self.label_names)
                wandb.termlog("Logged precision-recall curve.")
            finally:
                wandb.finish()

        self.probes[hook_name] = clf

    def train_all_probes(self, hook_names: List[str]):
        pbar = tqdm(hook_names, desc="Training probes")
        for hook_name in pbar:
            pbar.set_description(f"Training probe for {hook_name}")
            self.train_probe(hook_name)

    def save_models(self, save_dir: Path | str):
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for hook_name, model in self.probes.items():
            save_path = save_dir / f"{hook_name}_probe.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Saved probe for {hook_name} to {save_path}")
