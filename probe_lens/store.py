from typing import List, Union, Callable
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from jaxtyping import Float
from datasets import Dataset, DatasetDict, Features, Array2D, ClassLabel
import numpy as np


def _identity_filter(activations: np.ndarray) -> np.ndarray:
    return activations


class ActivationStore:
    hook_level = 9999

    def __init__(
        self,
        model: Union[HookedTransformer, SAE],
        hooks: Union[str, List[str] | HookPoint | List[HookPoint]],
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

        hook_names: list[str]
        hook_points: list[HookPoint]
        if isinstance(hooks, HookPoint):
            hook_names = [hooks.name]
            hook_points = [hooks]
        elif isinstance(hooks[0], HookPoint):
            hook_names = [hook.name for hook in hooks]
            hook_points = hooks
        elif isinstance(hooks, str):
            hook_names = [hooks]
            hook_points = [model.mod_dict[hooks]]
        elif isinstance(hooks[0], str):
            hook_names = hooks
            hook_points = [model.mod_dict[hook_name] for hook_name in hooks]
        else:
            raise ValueError(f"Invalid hook_names: {hooks}")

        self.class_names = class_names
        self.act_filter_fn = act_filter_fn
        self.class_labels = ClassLabel(
            num_classes=len(self.class_names), names=self.class_names
        )
        self.current_split: str = splits[0]
        self.dataset_dict = {
            split: {
                "activations": {hook: [] for hook in hook_names},
                "label": [],
            }
            for split in splits
        }

        for hook_name, hook_point in zip(hook_names, hook_points):
            model.check_and_add_hook(
                hook_point=hook_point,
                hook_point_name=hook_name,
                hook=self._hook_fn,
                level=self.hook_level,
            )

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
        self.dataset_dict[self.current_split]["label"].extend(labels)

    def compile_dataset(self) -> DatasetDict:
        """
        {
            train: {
                label: ClassLabel
                hook_name: Array[T, D]
            },
            test: {...}
        }
        """

        hf_dataset = DatasetDict()
        for split in self.dataset_dict:
            split_data = self.dataset_dict[split]
            labels = np.array(split_data["label"])
            hf_data = {}
            hf_data["label"] = labels
            features = Features({"label": self.class_labels})
            for hook_name in split_data["activations"]:
                activations = np.concatenate(
                    split_data["activations"][hook_name], axis=0
                )
                if activations.ndim != 3:
                    raise ValueError(
                        f"Activations are expected to be of shape (B, T, D). Given: {activations.shape}"
                    )
                if activations.shape[0] != labels.shape[0]:
                    raise ValueError(
                        "Activations and labels shape do not match! please pass the corresponding samples for each label"
                    )

                hf_data[hook_name] = activations
                # batch dimension is the rows of the dataset
                # Shape T, D
                features[hook_name] = Array2D(
                    shape=activations.shape[-2:], dtype="float32"
                )
            hf_dataset[split] = Dataset.from_dict(hf_data, features=features)
        return hf_dataset

    def detach(self):
        self.model.reset_hooks(level=self.hook_level)

    def _hook_fn(self, tensor, hook: HookPoint):
        assert hook.name is not None
        acts = tensor.detach().cpu().numpy()
        acts = self.act_filter_fn(acts)
        acts_store = self.dataset_dict[self.current_split]["activations"][hook.name]
        if len(acts_store):
            prev_act_dim = acts_store[0].shape[-1]
            if prev_act_dim != acts.shape[-1]:
                raise ValueError(
                    "Stored activations must all be same shape. Pad input or pass custom filter"
                )
        acts_store.append(acts)
        return tensor
