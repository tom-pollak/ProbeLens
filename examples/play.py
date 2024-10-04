# %%

import json
import torch
from sae_lens import SAE, HookedSAETransformer
from probe_lens import ActivationStore, ProbeTrainer

model = HookedSAETransformer.from_pretrained("gpt2-small")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb", sae_id="blocks.7.hook_resid_pre", device=device
)
hook_point = sae.cfg.hook_name

# %%
# ███████████████████████████  Capturing Activations  ████████████████████████████

def filter_fn(activations):
    return activations[:, -1:, :]

store = ActivationStore(
    model, hook_point, class_names=["negative", "positive"], act_filter_fn=filter_fn
)

with open("data/sentiment.json", "r") as file:
    data = json.load(file)
    prompts = data["prompts"]
    labels = data["labels"]

num_rows = len(prompts)
prompts_train, prompts_test = prompts[: num_rows // 2], prompts[num_rows // 2 :]
labels_train, labels_test = labels[: num_rows // 2], labels[num_rows // 2 :]

# Train
store.set_split("train")
store.add_labels(labels_train)

batch_size = 16
for i in range(0, len(prompts_train), batch_size):
    batch = prompts_train[i : i + batch_size]
    model(batch)

# Test
store.set_split("test")
store.add_labels(labels_test)

batch_size = 16
for i in range(0, len(prompts_test), batch_size):
    batch = prompts_test[i : i + batch_size]
    model(batch)

# Compile Dataset
dataset = store.compile_dataset()
store.detach()
print(dataset)

# %%
# ███████████████████████████  Training Probes  ████████████████████████████


probe_trainer = ProbeTrainer(
    dataset,
    flatten_T="batch",
    wandb_project="ProbeLens4",
    max_iter=1000,
)
probe_trainer.train()
probe_trainer.save_probes("./probes")

# %%
# ███████████████████████████  Visualizing Results  ████████████████████████████
import numpy as np
from probe_lens.viz import (
    visualize_model_weights,
    plot_activation_distribution,
    plot_probe_accuracy,
)

# Load a trained probe
# model_data = np.load("./probes/blocks.0.hook_resid_pre_probe.npy", allow_pickle=True).item()

# Visualize probe weights


visualize_model_weights(probe_trainer.probes[hook_point], hook_point)
# plot_activation_distribution(model_store.activations[hook_point], 0, hook_point)
plot_probe_accuracy(probe_trainer.history)
