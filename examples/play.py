# %%
# ███████████████████████████  Capturing Activations  ████████████████████████████

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
# model.add_sae(sae, hook_point)

model_store = ActivationStore(model, hook_point, t_slice=-1)
# sae_store = ActivationStore(sae, "hook_sae_acts_post")


# Load prompts and labels from data.json
with open("data/sentiment.json", "r") as file:
    data = json.load(file)
    prompts = data["prompts"]
    labels = data["labels"]

model_store.add_labels(labels)

batch_size = 16
for i in range(0, len(prompts), batch_size):
    batch = prompts[i : i + batch_size]
    model(batch)

# sae_store.add_labels(labels)

activation_dir = "./activations"
model_store.save_all(activation_dir)
# sae_store.save_all(activation_dir)

model_store.detach()
# sae_store.detach()

# %%
# ███████████████████████████  Training Probes  ████████████████████████████


probe_trainer = ProbeTrainer(
    model_store,
    wandb_project="ProbeLens3",
    label_names=["negative", "positive"],
    max_iter=1000,
)
probe_trainer.train_probe(hook_point)
probe_trainer.save_models("./probes")

# Similarly, initialize ProbeTrainer for SAE model hooks if needed
# sae_probe_trainer = ProbeTrainer(
#     sae_store,
#     project_name="ProbeLens_SAE",
#     metrics=[accuracy_score, precision_score, recall_score, f1_score],
#     model_class=LogisticRegression,
#     max_iter=1000
# )
# sae_probe_trainer.train_probe(hook_point)
# sae_probe_trainer.save_models("./probes_sae")

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
