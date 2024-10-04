from typing import List, Union, Callable
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from jaxtyping import Float
from datasets import Dataset, DatasetDict, Features, Array3D, ClassLabel
import numpy as np


def _identity_filter(activations: np.ndarray) -> np.ndarray:
    return activations


class ActivationStore:
    hook_level = 9999

    def __init__(
        self,
        model: Union[HookedTransformer, SAE],
        hook_names: Union[str, List[str]],
        class_names: list[str],
        act_filter_fn: Callable[
            [Float[np.ndarray, "B T D"]], Float[np.ndarray, "B T_p D"]
        ] = _identity_filter,
        splits=("train", "test"),
    ):
        """
        act_filter_fn: Select which activations to store
        splits: DatasetDict splits. `current_split` is initialized as first one
        """
        self.model = model
        self.hook_names = [hook_names] if isinstance(hook_names, str) else hook_names
        self.class_names = class_names
        self.act_filter_fn = act_filter_fn
        self.class_labels = ClassLabel(
            num_classes=len(self.class_names), names=self.class_names
        )
        self.current_split: str = splits[0]
        self.dataset_dict = {
            split: {
                "activations": {hook: [] for hook in self.hook_names},
                "labels": [],
            }
            for split in splits
        }

        for hook_name in self.hook_names:
            self.model.add_hook(hook_name, self._hook_fn, level=self.hook_level)

    def set_split(self, split: str):
        """
        Set the current dataset split (e.g., 'train', 'test').
        """
        if split not in self.dataset_dict:
            raise ValueError(
                f"Invalid split name: {split}. Choose from 'train', 'val', 'test'."
            )
        self.current_split = split

    def add_labels(self, labels: List[int]):
        """
        Add labels for a specific split. Defaults to current split.
        """
        if self.current_split not in self.dataset_dict:
            raise ValueError(
                f"Invalid split name: {self.current_split}. Choose from 'train', 'val', 'test'."
            )
        self.dataset_dict[self.current_split]["labels"].extend(labels)

    def compile(self):
        self._detach()

        hf_dataset = DatasetDict()
        for split in self.dataset_dict:
            print(f"Split: {split}")
            split_data = self.dataset_dict[split]
            split_data["labels"] = np.array(split_data["labels"])
            features = Features(
                {
                    "activations": {},
                    "labels": self.class_labels,
                }
            )
            print(split_data)
            for hook_name in split_data["activations"]:
                # Stack activations [np.ndarray] => np.ndarray
                activations = np.concatenate(
                    split_data["activations"][hook_name], axis=0
                )
                if activations.ndim != 3:
                    raise ValueError(
                        f"Activations are expected to be of shape (B, T, D). Given: {activations.shape}"
                    )
                split_data["activations"][hook_name] = activations

                # Cast dataset into efficient format
                features["activations"][hook_name] = Array3D(
                    shape=activations.shape, dtype="float32"
                )

                print(
                    f"Activations shape for {hook_name}: {np.array(split_data['activations'][hook_name]).shape}"
                )

            print(f"Labels count: {len(split_data['labels'])}")
            print("Features:", features)

            hf_dataset[split] = Dataset.from_dict(split_data, features=features)

        print("dataset compiled")
        return hf_dataset

    def _detach(self):
        self.model.reset_hooks(level=self.hook_level)

    def _hook_fn(self, tensor, hook: HookPoint):
        assert hook.name is not None
        acts = tensor.detach().cpu().numpy()
        acts = self.act_filter_fn(acts)
        self.dataset_dict[self.current_split]["activations"][hook.name].append(acts)  # type: ignore
        return tensor
