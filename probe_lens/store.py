from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Union, Optional
import numpy as np
from jaxtyping import Float

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE


class ActivationStore:
    hook_level = 9999

    def __init__(
        self,
        model: Union[HookedTransformer, SAE],
        hook_names: Union[str, List[str]],
        t_slice: Optional[Union[int, slice]] = None,
    ):
        self.model = model
        self.hook_names = [hook_names] if isinstance(hook_names, str) else hook_names
        self.activations: Dict[str, List[Float[np.ndarray, "B T D"]]] = defaultdict(list)  # fmt: off
        self.labels: list[int] = []

        self.t_slice = (
            slice(t_slice, t_slice + 1 if t_slice != -1 else None)
            if isinstance(t_slice, int)
            else t_slice
        )

        for hook_name in self.hook_names:
            self.model.add_hook(hook_name, self.hook_fn, level=self.hook_level)

    def hook_fn(
        self, tensor: Float[torch.Tensor, "B T D"], hook: HookPoint
    ) -> Float[torch.Tensor, "B T D"]:
        acts_np = tensor.detach().cpu().numpy()

        if self.t_slice is not None:
            acts_np = acts_np[:, self.t_slice, :]  # apply to T

        assert hook.name is not None
        self.activations[hook.name].append(acts_np)
        return tensor

    def detach(self):
        self.model.reset_hooks(level=self.hook_level)

    def add_labels(self, labels: list[int]):
        self.labels.extend(labels)

    def save_all(self, save_dir: Union[Path, str]):
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for hook_name in self.hook_names:
            assert (
                hook_name in self.activations
            ), f"No activations for hook: {hook_name}"
            if not self.activations[hook_name]:
                continue

            stacked_acts = np.concatenate(self.activations[hook_name], axis=0)
            act_path = save_dir / f"{hook_name}_activations.npy"
            np.save(act_path, stacked_acts)
            print(f"Saved activations for {hook_name} to {act_path}")

        if self.labels:
            labels_path = save_dir / "labels.npy"
            np.save(labels_path, np.array(self.labels))
            print(f"Saved labels to {labels_path}")

    @classmethod
    def load_all(
        cls, model: Union[HookedTransformer, SAE], directory: Union[Path, str]
    ):
        if isinstance(directory, str):
            directory = Path(directory)
        hook_names = [p.stem.split("_")[0] for p in directory.glob("*_activations.npy")]
        hook_handler = cls(model, hook_names)
        for hook_name in hook_names:
            activation_path = directory / f"{hook_name}_activations.npy"
            assert activation_path.exists()
            hook_handler.activations[hook_name] = np.load(activation_path)
            print(f"Loaded activations for {hook_name} from {activation_path}")

            labels_path = directory / "labels.npy"
            if labels_path.exists():
                hook_handler.labels = np.load(labels_path).tolist()
                print(f"Loaded labels for {hook_name} from {labels_path}")

        return hook_handler
