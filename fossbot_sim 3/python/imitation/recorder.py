"""
imitation/recorder.py
=====================
Chapter 13 — Imitation Learning & Demonstration Recording

Components:
  - DemonstrationRecorder  : records (obs, action) pairs during any episode
  - DemonstrationDataset   : loads recorded files into a PyTorch dataset
  - BehavioralCloningTrainer: trains a policy via supervised learning on demos
  - BCPolicyWrapper        : wraps a trained BC model for use in train_sb3.py

Usage — Recording:
    recorder = DemonstrationRecorder(save_dir="demos/")
    recorder.start("maze_seed42")
    for step in range(500):
        obs = env.get_obs()
        action = my_scripted_policy(obs)
        state, reward, done, _, _ = env.step(action)
        recorder.record(obs, action, state)
        if done: break
    recorder.stop()

Usage — Training:
    trainer = BehavioralCloningTrainer(demo_dir="demos/", obs_dim=16, action_dim=2)
    trainer.train(epochs=50, batch_size=64)
    trainer.save("logs/bc_policy.pt")

Usage — SB3 Integration:
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, ...)
    trainer.load_into_sb3(model)   # initialise policy weights from BC
    model.learn(total_timesteps=100_000)   # fine-tune with RL
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ─────────────────────────────────────────────
#  DEMONSTRATION RECORDER
# ─────────────────────────────────────────────

class DemonstrationRecorder:
    """
    Records (observation, action, state_label) tuples during any
    control session — human, scripted policy, or DWA planner.
    Writes one JSON-Lines file per recording session.
    Chapter 13.1
    """

    def __init__(self, save_dir: str = "demos/"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._session_id = ""
        self._step = 0
        self._active = False

    def start(self, session_id: str = "") -> None:
        """Begin a new recording session."""
        self._session_id = session_id or time.strftime("demo_%Y%m%d_%H%M%S")
        path = self.save_dir / f"{self._session_id}.jsonl"
        self._file = open(path, "w", encoding="utf-8")
        self._step = 0
        self._active = True
        print(f"[DemoRecorder] Recording → {path}")

    def record(
        self,
        obs: np.ndarray,
        action: Dict[str, float],
        state: Dict[str, Any]
    ) -> None:
        """Record one transition. Call after every env.step()."""
        if not self._active or self._file is None:
            return

        entry = {
            "step":        self._step,
            "timestamp":   state.get("timestamp", 0.0),
            "obs":         obs.tolist(),
            "action":      [action.get("throttle", 0.0), action.get("steer", 0.0)],
            "state_label": state.get("state_label", ""),
            "success_hint": False   # filled in at stop() if goal reached
        }
        self._file.write(json.dumps(entry) + "\n")
        self._step += 1

    def stop(self, success: bool = False) -> str:
        """Close the recording. Returns path to saved file."""
        if self._file is None:
            return ""

        # If success, mark last 20% of steps as success_hint
        # (useful for weighted BC loss)
        self._file.close()
        if success:
            self._mark_success_hints()

        path = str(self.save_dir / f"{self._session_id}.jsonl")
        print(f"[DemoRecorder] Saved {self._step} steps → {path}")
        self._active = False
        self._file = None
        return path

    def _mark_success_hints(self) -> None:
        path = self.save_dir / f"{self._session_id}.jsonl"
        lines = path.read_text().strip().split("\n")
        n = len(lines)
        mark_from = int(n * 0.8)
        updated = []
        for i, line in enumerate(lines):
            d = json.loads(line)
            if i >= mark_from:
                d["success_hint"] = True
            updated.append(json.dumps(d))
        path.write_text("\n".join(updated) + "\n")

    @property
    def is_active(self) -> bool:
        return self._active


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────

class DemonstrationDataset:
    """
    Loads all .jsonl demo files from a directory into numpy arrays.
    Compatible with PyTorch DataLoader.
    """

    def __init__(self, demo_dir: str, obs_dim: int = 16, action_dim: int = 2):
        if not _TORCH_OK:
            raise ImportError("pip install torch")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observations, self.actions = self._load(Path(demo_dir))
        print(f"[DemoDataset] Loaded {len(self.observations)} transitions from {demo_dir}")

    def _load(self, demo_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        all_obs, all_act = [], []
        files = list(demo_dir.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {demo_dir}")
        for f in files:
            for line in f.read_text().strip().split("\n"):
                if not line:
                    continue
                d = json.loads(line)
                all_obs.append(d["obs"])
                all_act.append(d["action"])
        return np.array(all_obs, dtype=np.float32), np.array(all_act, dtype=np.float32)

    def as_torch_dataset(self):
        if not _TORCH_OK:
            raise ImportError("pip install torch")
        return _TorchDemoDataset(self.observations, self.actions)


class _TorchDemoDataset(Dataset if _TORCH_OK else object):
    def __init__(self, obs, acts):
        self.obs  = torch.tensor(obs,  dtype=torch.float32)
        self.acts = torch.tensor(acts, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]


# ─────────────────────────────────────────────
#  BEHAVIORAL CLONING TRAINER
# ─────────────────────────────────────────────

class BehavioralCloningTrainer:
    """
    Chapter 13 — Trains a policy via Behavioral Cloning (supervised learning).
    Architecture matches SB3 MlpPolicy actor network so weights can transfer.
    """

    def __init__(
        self,
        demo_dir: str,
        obs_dim: int = 16,
        action_dim: int = 2,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        device: str = "auto"
    ):
        if not _TORCH_OK:
            raise ImportError("pip install torch")

        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build network matching SB3 MlpPolicy actor
        layers = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers += [nn.Linear(in_size, action_dim), nn.Tanh()]  # Tanh → [-1,1]
        self.policy = nn.Sequential(*layers).to(self.device)

        # Load data
        dataset = DemonstrationDataset(demo_dir, obs_dim, action_dim)
        self.dataloader = DataLoader(
            dataset.as_torch_dataset(),
            batch_size=64,
            shuffle=True,
            drop_last=True
        )

        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.loss_fn   = nn.MSELoss()
        self._train_losses: List[float] = []

    def train(self, epochs: int = 50, verbose: bool = True) -> List[float]:
        """Run BC training. Returns list of per-epoch mean losses."""
        print(f"[BCTrainer] Training for {epochs} epochs on {self.device}")
        self.policy.train()

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for obs_batch, act_batch in self.dataloader:
                obs_batch = obs_batch.to(self.device)
                act_batch = act_batch.to(self.device)

                pred = self.policy(obs_batch)
                loss = self.loss_fn(pred, act_batch)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                epoch_losses.append(loss.item())

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            self._train_losses.append(mean_loss)

            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{epochs}  loss={mean_loss:.5f}")

        print(f"[BCTrainer] Done. Final loss: {self._train_losses[-1]:.5f}")
        return self._train_losses

    def save(self, path: str) -> None:
        """Save policy weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        print(f"[BCTrainer] Policy saved → {path}")

    def load(self, path: str) -> None:
        """Load policy weights."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[BCTrainer] Policy loaded ← {path}")

    def load_into_sb3(self, sb3_model) -> None:
        """
        Transfer BC policy weights into an SB3 model's actor network.
        Call this before sb3_model.learn() to warm-start RL with BC weights.
        Chapter 13 — BC → RL Pre-training Integration.
        """
        sb3_actor = sb3_model.policy.mlp_extractor.policy_net
        bc_layers = [l for l in self.policy if isinstance(l, nn.Linear)]
        sb3_linear_layers = [l for l in sb3_actor if isinstance(l, nn.Linear)]

        transferred = 0
        for bc_l, sb3_l in zip(bc_layers, sb3_linear_layers):
            if bc_l.weight.shape == sb3_l.weight.shape:
                sb3_l.weight.data.copy_(bc_l.weight.data)
                sb3_l.bias.data.copy_(bc_l.bias.data)
                transferred += 1

        print(f"[BCTrainer] Transferred {transferred} layers into SB3 policy.")

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run inference. Returns action array."""
        self.policy.eval()
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            out = self.policy(t).cpu().numpy()[0]
        return out
