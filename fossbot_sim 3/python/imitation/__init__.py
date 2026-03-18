"""
imitation package — Chapter 13
================================
Imitation Learning & Demonstration Recording

    from imitation.recorder import DemonstrationRecorder, BehavioralCloningTrainer
"""
from .recorder import DemonstrationRecorder, DemonstrationDataset, BehavioralCloningTrainer
__all__ = ["DemonstrationRecorder", "DemonstrationDataset", "BehavioralCloningTrainer"]
