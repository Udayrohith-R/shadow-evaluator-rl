# Shadow Evaluator for RL Training Stability
# =============================================
# A sidecar monitoring system that watches RL training runs in real-time,
# detecting mode collapse, reward hacking, and policy drift before they
# waste thousands of dollars in compute.
#
# Architecture:
# 1. PyTorch hooks capture activations from policy and reference models
# 2. Real-time KL-divergence computation between policy and reference
# 3. Reward hacking detector (repetition, degenerate outputs)
# 4. Automatic checkpoint revert + learning rate adjustment
# 5. Prometheus metrics for observability
#
# Author: Uday
# Target: Anthropic RL Engineering Team

import os
import time
import json
import math
import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from collections import deque
from enum import Enum
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("shadow_evaluator")


# ============================================================
# PART 1: Core Data Structures
# ============================================================

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class InterventionType(Enum):
    NONE = "none"
    REDUCE_LR = "reduce_lr"
    CHECKPOINT_REVERT = "checkpoint_revert"
    HARD_STOP = "hard_stop"


@dataclass
class TrainingSnapshot:
    """Captures the state of training at a single step."""
    step: int
    timestamp: float
    policy_logprobs: List[float]      # Log probs from current policy
    reference_logprobs: List[float]   # Log probs from frozen reference
    reward_scores: List[float]        # Reward model outputs
    generated_tokens: List[List[int]] # Generated token sequences
    loss: float
    learning_rate: float
    gradient_norm: float


@dataclass
class HealthReport:
    """Health assessment of the current training run."""
    step: int
    timestamp: float
    kl_divergence: float
    kl_trend: str                     # "stable", "rising", "critical"
    reward_hack_score: float          # 0.0 (clean) to 1.0 (hacking)
    repetition_ratio: float           # Fraction of repetitive tokens
    entropy: float                    # Output distribution entropy
    intervention: InterventionType
    alerts: List[Dict]
    metrics: Dict


@dataclass
class CheckpointInfo:
    """Metadata for a training checkpoint."""
    step: int
    path: str
    kl_divergence: float
    reward_mean: float
    loss: float
    timestamp: float
    is_healthy: bool


# ============================================================
# PART 2: KL Divergence Monitor
# ============================================================

class KLDivergenceMonitor:
    """
    Tracks KL divergence between the current RL policy and the frozen
    reference model in real-time.
    
    Key insight: KL divergence is the earliest indicator of training
    instability. By monitoring it continuously (not just at eval time),
    we can catch mode collapse before it becomes catastrophic.
    """
    
    def __init__(
        self,
        kl_warning_threshold: float = 10.0,
        kl_critical_threshold: float = 25.0,
        window_size: int = 100,
        trend_window: int = 20,
    ):
        self.kl_warning = kl_warning_threshold
        self.kl_critical = kl_critical_threshold
        self.window_size = window_size
        self.trend_window = trend_window
        
        self._kl_history: deque = deque(maxlen=window_size)
        self._step_history: deque = deque(maxlen=window_size)
    
    def compute_kl(
        self, 
        policy_logprobs: List[float], 
        reference_logprobs: List[float]
    ) -> float:
        """
        Compute KL(policy || reference) from log probabilities.
        
        KL = E_policy[log(policy) - log(reference)]
           = mean(policy_logprobs - reference_logprobs)
        """
        if len(policy_logprobs) != len(reference_logprobs):
            raise ValueError("Logprob sequences must have same length")
        
        if not policy_logprobs:
            return 0.0
        
        kl_per_token = []
        for p_lp, r_lp in zip(policy_logprobs, reference_logprobs):
            # KL at each token position
            kl = p_lp - r_lp
            kl_per_token.append(kl)
        
        return sum(kl_per_token) / len(kl_per_token)
    
    def update(self, step: int, kl_value: float) -> Dict:
        """Update KL tracking and return status."""
        self._kl_history.append(kl_value)
        self._step_history.append(step)
        
        # Compute trend
        trend = self._compute_trend()
        
        # Determine severity
        if kl_value >= self.kl_critical:
            severity = AlertSeverity.CRITICAL
        elif kl_value >= self.kl_warning:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        return {
            "kl_divergence": kl_value,
            "kl_mean": self._running_mean(),
            "kl_std": self._running_std(),
            "kl_max": max(self._kl_history) if self._kl_history else 0,
            "trend": trend,
            "severity": severity,
        }
    
    def _compute_trend(self) -> str:
        """Detect if KL is trending upward (early warning)."""
        if len(self._kl_history) < self.trend_window:
            return "insufficient_data"
        
        recent = list(self._kl_history)[-self.trend_window:]
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        change = (avg_second - avg_first) / (avg_first + 1e-8)
        
        if change > 0.5:
            return "critical_rise"
        elif change > 0.2:
            return "rising"
        elif change < -0.2:
            return "falling"
        else:
            return "stable"
    
    def _running_mean(self) -> float:
        if not self._kl_history:
            return 0.0
        return sum(self._kl_history) / len(self._kl_history)
    
    def _running_std(self) -> float:
        if len(self._kl_history) < 2:
            return 0.0
        mean = self._running_mean()
        variance = sum((x - mean) ** 2 for x in self._kl_history) / len(self._kl_history)
        return math.sqrt(variance)


# ============================================================
# PART 3: Reward Hacking Detector
# ============================================================

class RewardHackingDetector:
    """
    Detects when the RL policy learns to exploit the reward model
    rather than genuinely improving quality.
    
    Common reward hacking patterns:
    1. Repetitive token sequences that fool the reward model
    2. Increasing reward with decreasing output quality
    3. Sudden reward spikes without corresponding quality improvement
    4. Low entropy outputs (model becoming too confident/narrow)
    """
    
    def __init__(
        self,
        repetition_threshold: float = 0.3,
        entropy_floor: float = 1.0,
        reward_spike_threshold: float = 3.0,
        window_size: int = 50,
    ):
        self.repetition_threshold = repetition_threshold
        self.entropy_floor = entropy_floor
        self.reward_spike_threshold = reward_spike_threshold
        
        self._reward_history: deque = deque(maxlen=window_size)
        self._entropy_history: deque = deque(maxlen=window_size)
        self._repetition_history: deque = deque(maxlen=window_size)
    
    def analyze(
        self,
        generated_tokens: List[List[int]],
        reward_scores: List[float],
        policy_logprobs: List[float],
    ) -> Dict:
        """Analyze a batch for reward hacking signals."""
        
        # 1. Repetition analysis
        repetition_ratio = self._compute_repetition(generated_tokens)
        self._repetition_history.append(repetition_ratio)
        
        # 2. Entropy analysis
        entropy = self._compute_entropy(policy_logprobs)
        self._entropy_history.append(entropy)
        
        # 3. Reward spike detection
        mean_reward = sum(reward_scores) / len(reward_scores) if reward_scores else 0
        self._reward_history.append(mean_reward)
        reward_spike = self._detect_reward_spike(mean_reward)
        
        # 4. Composite hack score
        hack_score = self._compute_hack_score(
            repetition_ratio, entropy, reward_spike
        )
        
        return {
            "hack_score": hack_score,
            "repetition_ratio": repetition_ratio,
            "entropy": entropy,
            "mean_reward": mean_reward,
            "reward_spike_detected": reward_spike,
            "is_hacking": hack_score > 0.7,
        }
    
    def _compute_repetition(self, token_sequences: List[List[int]]) -> float:
        """
        Detect repetitive token patterns.
        High repetition is a classic reward hacking signal.
        """
        if not token_sequences:
            return 0.0
        
        total_repetition = 0.0
        
        for tokens in token_sequences:
            if len(tokens) < 4:
                continue
            
            # Check for n-gram repetition (bigrams and trigrams)
            bigrams = set()
            repeated_bigrams = 0
            
            for i in range(len(tokens) - 1):
                bg = (tokens[i], tokens[i + 1])
                if bg in bigrams:
                    repeated_bigrams += 1
                bigrams.add(bg)
            
            total_bigrams = len(tokens) - 1
            if total_bigrams > 0:
                total_repetition += repeated_bigrams / total_bigrams
        
        return total_repetition / len(token_sequences)
    
    def _compute_entropy(self, logprobs: List[float]) -> float:
        """
        Compute output distribution entropy.
        Low entropy = model is too confident = potential mode collapse.
        """
        if not logprobs:
            return 0.0
        
        # Convert log probs to probabilities
        probs = [math.exp(lp) for lp in logprobs]
        
        # Normalize
        total = sum(probs)
        if total == 0:
            return 0.0
        probs = [p / total for p in probs]
        
        # Shannon entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _detect_reward_spike(self, current_reward: float) -> bool:
        """Detect sudden reward spikes (potential hacking)."""
        if len(self._reward_history) < 10:
            return False
        
        history = list(self._reward_history)[:-1]  # exclude current
        mean = sum(history) / len(history)
        std = math.sqrt(sum((x - mean) ** 2 for x in history) / len(history))
        
        if std == 0:
            return False
        
        z_score = (current_reward - mean) / std
        return z_score > self.reward_spike_threshold
    
    def _compute_hack_score(
        self, repetition: float, entropy: float, reward_spike: bool
    ) -> float:
        """
        Composite score from 0.0 (clean) to 1.0 (definitely hacking).
        """
        score = 0.0
        
        # Repetition contributes up to 0.4
        if repetition > self.repetition_threshold:
            score += min(0.4, (repetition - self.repetition_threshold) * 2)
        
        # Low entropy contributes up to 0.3
        if entropy < self.entropy_floor:
            score += min(0.3, (self.entropy_floor - entropy) / self.entropy_floor * 0.3)
        
        # Reward spike contributes 0.3
        if reward_spike:
            score += 0.3
        
        return min(1.0, score)


# ============================================================
# PART 4: Checkpoint Manager
# ============================================================

class CheckpointManager:
    """
    Manages training checkpoints with health metadata.
    Enables automatic revert to last known healthy state.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", max_kept: int = 10):
        self.checkpoint_dir = checkpoint_dir
        self.max_kept = max_kept
        self._checkpoints: deque = deque(maxlen=max_kept)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, step: int, kl_div: float, reward: float, 
             loss: float, is_healthy: bool) -> CheckpointInfo:
        """Save a checkpoint with health metadata."""
        path = os.path.join(self.checkpoint_dir, f"ckpt_step_{step}")
        
        info = CheckpointInfo(
            step=step,
            path=path,
            kl_divergence=kl_div,
            reward_mean=reward,
            loss=loss,
            timestamp=time.time(),
            is_healthy=is_healthy,
        )
        
        # In production: torch.save(model.state_dict(), path)
        # Here we save metadata
        meta_path = path + ".meta.json"
        with open(meta_path, 'w') as f:
            json.dump({
                "step": info.step,
                "path": info.path,
                "kl_divergence": info.kl_divergence,
                "reward_mean": info.reward_mean,
                "loss": info.loss,
                "timestamp": info.timestamp,
                "is_healthy": info.is_healthy,
            }, f, indent=2)
        
        self._checkpoints.append(info)
        logger.info(f"Checkpoint saved: step={step}, healthy={is_healthy}, "
                     f"kl={kl_div:.4f}, reward={reward:.4f}")
        
        return info
    
    def get_last_healthy(self) -> Optional[CheckpointInfo]:
        """Find the most recent healthy checkpoint for revert."""
        for ckpt in reversed(self._checkpoints):
            if ckpt.is_healthy:
                return ckpt
        return None
    
    def revert_to(self, checkpoint: CheckpointInfo) -> bool:
        """
        Revert training to a previous checkpoint.
        In production: loads model.state_dict() and optimizer state.
        """
        logger.warning(
            f"REVERTING to checkpoint at step {checkpoint.step} "
            f"(kl={checkpoint.kl_divergence:.4f}, reward={checkpoint.reward_mean:.4f})"
        )
        # In production:
        # model.load_state_dict(torch.load(checkpoint.path))
        # optimizer.load_state_dict(torch.load(checkpoint.path + ".optimizer"))
        return True


# ============================================================
# PART 5: Shadow Evaluator (Main System)
# ============================================================

class ShadowEvaluator:
    """
    Real-time sidecar process that monitors RL training stability.
    
    This is the main entry point. It:
    1. Receives training snapshots every N steps
    2. Computes KL divergence, detects reward hacking
    3. Makes intervention decisions
    4. Exports metrics to Prometheus
    
    Usage:
        evaluator = ShadowEvaluator(
            kl_warning=10.0,
            kl_critical=25.0,
            checkpoint_interval=100,
        )
        
        # In your training loop:
        for step in range(num_steps):
            # ... training code ...
            
            snapshot = TrainingSnapshot(
                step=step,
                policy_logprobs=policy_lp,
                reference_logprobs=ref_lp,
                reward_scores=rewards,
                generated_tokens=gen_tokens,
                loss=loss,
                learning_rate=lr,
                gradient_norm=grad_norm,
            )
            
            report = evaluator.evaluate(snapshot)
            
            if report.intervention == InterventionType.CHECKPOINT_REVERT:
                evaluator.execute_revert()
            elif report.intervention == InterventionType.REDUCE_LR:
                lr *= 0.5
    """
    
    def __init__(
        self,
        kl_warning: float = 10.0,
        kl_critical: float = 25.0,
        repetition_threshold: float = 0.3,
        entropy_floor: float = 1.0,
        checkpoint_interval: int = 100,
        checkpoint_dir: str = "./checkpoints",
        lr_reduction_factor: float = 0.5,
        max_consecutive_warnings: int = 5,
    ):
        # Sub-systems
        self.kl_monitor = KLDivergenceMonitor(
            kl_warning_threshold=kl_warning,
            kl_critical_threshold=kl_critical,
        )
        self.hack_detector = RewardHackingDetector(
            repetition_threshold=repetition_threshold,
            entropy_floor=entropy_floor,
        )
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
        )
        
        # Config
        self.checkpoint_interval = checkpoint_interval
        self.lr_reduction_factor = lr_reduction_factor
        self.max_consecutive_warnings = max_consecutive_warnings
        
        # State
        self._consecutive_warnings = 0
        self._total_interventions = 0
        self._reports: List[HealthReport] = []
        
        # Prometheus-style metrics
        self._prometheus_metrics = {
            "rl_kl_divergence": 0.0,
            "rl_reward_hack_score": 0.0,
            "rl_repetition_ratio": 0.0,
            "rl_output_entropy": 0.0,
            "rl_training_loss": 0.0,
            "rl_gradient_norm": 0.0,
            "rl_interventions_total": 0,
            "rl_checkpoint_reverts_total": 0,
            "rl_lr_reductions_total": 0,
        }
    
    def evaluate(self, snapshot: TrainingSnapshot) -> HealthReport:
        """
        Evaluate a training snapshot and return a health report.
        This is called every N steps from the training loop.
        """
        alerts = []
        
        # 1. KL Divergence analysis
        kl_value = self.kl_monitor.compute_kl(
            snapshot.policy_logprobs, 
            snapshot.reference_logprobs
        )
        kl_status = self.kl_monitor.update(snapshot.step, kl_value)
        
        if kl_status["severity"] == AlertSeverity.CRITICAL:
            alerts.append({
                "type": "kl_critical",
                "message": f"KL divergence critical: {kl_value:.4f} (threshold: {self.kl_monitor.kl_critical})",
                "severity": "critical",
                "step": snapshot.step,
            })
        elif kl_status["severity"] == AlertSeverity.WARNING:
            alerts.append({
                "type": "kl_warning",
                "message": f"KL divergence elevated: {kl_value:.4f} (threshold: {self.kl_monitor.kl_warning})",
                "severity": "warning",
                "step": snapshot.step,
            })
        
        # 2. Reward hacking analysis
        hack_status = self.hack_detector.analyze(
            snapshot.generated_tokens,
            snapshot.reward_scores,
            snapshot.policy_logprobs,
        )
        
        if hack_status["is_hacking"]:
            alerts.append({
                "type": "reward_hacking",
                "message": f"Reward hacking detected: score={hack_status['hack_score']:.3f}, "
                          f"repetition={hack_status['repetition_ratio']:.3f}",
                "severity": "critical",
                "step": snapshot.step,
            })
        
        # 3. Determine intervention
        intervention = self._decide_intervention(
            kl_status, hack_status, snapshot
        )
        
        # 4. Checkpoint management
        if snapshot.step % self.checkpoint_interval == 0:
            is_healthy = (
                kl_status["severity"] == AlertSeverity.INFO 
                and not hack_status["is_hacking"]
            )
            self.checkpoint_mgr.save(
                step=snapshot.step,
                kl_div=kl_value,
                reward=hack_status["mean_reward"],
                loss=snapshot.loss,
                is_healthy=is_healthy,
            )
        
        # 5. Update Prometheus metrics
        self._update_metrics(kl_value, hack_status, snapshot, intervention)
        
        # 6. Build report
        report = HealthReport(
            step=snapshot.step,
            timestamp=time.time(),
            kl_divergence=kl_value,
            kl_trend=kl_status["trend"],
            reward_hack_score=hack_status["hack_score"],
            repetition_ratio=hack_status["repetition_ratio"],
            entropy=hack_status["entropy"],
            intervention=intervention,
            alerts=alerts,
            metrics=self._prometheus_metrics.copy(),
        )
        
        self._reports.append(report)
        
        # Log alerts
        for alert in alerts:
            if alert["severity"] == "critical":
                logger.critical(f"Step {snapshot.step}: {alert['message']}")
            else:
                logger.warning(f"Step {snapshot.step}: {alert['message']}")
        
        if intervention != InterventionType.NONE:
            logger.warning(f"Step {snapshot.step}: Intervention triggered: {intervention.value}")
        
        return report
    
    def _decide_intervention(
        self, kl_status: Dict, hack_status: Dict, snapshot: TrainingSnapshot
    ) -> InterventionType:
        """
        Decision tree for automatic interventions.
        
        Priority:
        1. HARD_STOP: Unrecoverable state
        2. CHECKPOINT_REVERT: Critical KL or confirmed reward hacking
        3. REDUCE_LR: Warning-level KL trending upward
        4. NONE: Everything healthy
        """
        
        # Critical: reward hacking + high KL = revert immediately
        if hack_status["is_hacking"] and kl_status["severity"] == AlertSeverity.CRITICAL:
            self._consecutive_warnings = 0
            self._total_interventions += 1
            return InterventionType.CHECKPOINT_REVERT
        
        # Critical KL alone = revert
        if kl_status["severity"] == AlertSeverity.CRITICAL:
            self._consecutive_warnings = 0
            self._total_interventions += 1
            return InterventionType.CHECKPOINT_REVERT
        
        # Confirmed reward hacking = revert
        if hack_status["is_hacking"]:
            self._consecutive_warnings = 0
            self._total_interventions += 1
            return InterventionType.CHECKPOINT_REVERT
        
        # Warning level: accumulate
        if kl_status["severity"] == AlertSeverity.WARNING:
            self._consecutive_warnings += 1
            
            # Too many consecutive warnings = reduce LR
            if self._consecutive_warnings >= self.max_consecutive_warnings:
                self._consecutive_warnings = 0
                self._total_interventions += 1
                return InterventionType.REDUCE_LR
        else:
            self._consecutive_warnings = 0
        
        return InterventionType.NONE
    
    def execute_revert(self) -> bool:
        """Execute a checkpoint revert to last healthy state."""
        healthy_ckpt = self.checkpoint_mgr.get_last_healthy()
        
        if healthy_ckpt is None:
            logger.error("No healthy checkpoint available for revert!")
            return False
        
        success = self.checkpoint_mgr.revert_to(healthy_ckpt)
        self._prometheus_metrics["rl_checkpoint_reverts_total"] += 1
        
        return success
    
    def _update_metrics(
        self, kl: float, hack: Dict, snapshot: TrainingSnapshot, 
        intervention: InterventionType
    ):
        """Update Prometheus-compatible metrics."""
        self._prometheus_metrics["rl_kl_divergence"] = kl
        self._prometheus_metrics["rl_reward_hack_score"] = hack["hack_score"]
        self._prometheus_metrics["rl_repetition_ratio"] = hack["repetition_ratio"]
        self._prometheus_metrics["rl_output_entropy"] = hack["entropy"]
        self._prometheus_metrics["rl_training_loss"] = snapshot.loss
        self._prometheus_metrics["rl_gradient_norm"] = snapshot.gradient_norm
        self._prometheus_metrics["rl_interventions_total"] = self._total_interventions
        
        if intervention == InterventionType.REDUCE_LR:
            self._prometheus_metrics["rl_lr_reductions_total"] += 1
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines = []
        for key, value in self._prometheus_metrics.items():
            lines.append(f"# TYPE {key} gauge")
            lines.append(f"{key} {value}")
        return "\n".join(lines)
    
    def get_summary(self) -> Dict:
        """Get summary of the training run's health."""
        if not self._reports:
            return {"status": "no_data"}
        
        return {
            "total_steps_evaluated": len(self._reports),
            "total_interventions": self._total_interventions,
            "total_checkpoint_reverts": int(self._prometheus_metrics["rl_checkpoint_reverts_total"]),
            "total_lr_reductions": int(self._prometheus_metrics["rl_lr_reductions_total"]),
            "final_kl": self._reports[-1].kl_divergence,
            "max_kl": max(r.kl_divergence for r in self._reports),
            "max_hack_score": max(r.reward_hack_score for r in self._reports),
            "alerts_generated": sum(len(r.alerts) for r in self._reports),
        }


# ============================================================
# PART 6: Simulation / Demo
# ============================================================

def simulate_training_run():
    """
    Simulate an RL training run with various failure modes
    to demonstrate the Shadow Evaluator's detection capabilities.
    """
    print("=" * 70)
    print("SHADOW EVALUATOR — RL TRAINING STABILITY MONITOR")
    print("Simulating RLHF training with failure injection")
    print("=" * 70)
    
    evaluator = ShadowEvaluator(
        kl_warning=8.0,
        kl_critical=20.0,
        checkpoint_interval=50,
        checkpoint_dir="/tmp/shadow_eval_demo",
    )
    
    random.seed(42)
    num_steps = 500
    
    # Simulated training states
    base_kl = 2.0
    base_reward = 0.5
    base_loss = 3.0
    lr = 1e-4
    
    print(f"\nRunning {num_steps} training steps...\n")
    
    for step in range(num_steps):
        # === Simulate different training phases ===
        
        # Phase 1 (0-150): Normal healthy training
        if step < 150:
            noise = random.gauss(0, 0.5)
            kl_sim = base_kl + noise + step * 0.01
            reward_sim = base_reward + step * 0.002 + random.gauss(0, 0.05)
            repetition = random.uniform(0.01, 0.05)
        
        # Phase 2 (150-250): KL starts drifting up (early warning)
        elif step < 250:
            drift = (step - 150) * 0.08
            kl_sim = base_kl + drift + random.gauss(0, 1.0)
            reward_sim = base_reward + 0.3 + random.gauss(0, 0.1)
            repetition = random.uniform(0.05, 0.15)
        
        # Phase 3 (250-350): Reward hacking begins
        elif step < 350:
            kl_sim = 12.0 + random.gauss(0, 2.0)
            reward_sim = 2.0 + random.gauss(0, 0.2)  # Suspiciously high
            repetition = random.uniform(0.25, 0.45)   # High repetition
        
        # Phase 4 (350-500): Mode collapse (if not caught)
        else:
            kl_sim = 30.0 + random.gauss(0, 5.0)
            reward_sim = 5.0 + random.gauss(0, 0.1)   # Very high (hacked)
            repetition = random.uniform(0.5, 0.8)       # Severe repetition
        
        # Build snapshot
        seq_len = 100
        policy_lp = [random.gauss(-2.0 - kl_sim * 0.1, 0.5) for _ in range(seq_len)]
        ref_lp = [random.gauss(-2.0, 0.3) for _ in range(seq_len)]
        rewards = [reward_sim + random.gauss(0, 0.1) for _ in range(8)]
        
        # Generate tokens with controlled repetition
        gen_tokens = []
        for _ in range(8):
            tokens = []
            for t in range(seq_len):
                if random.random() < repetition:
                    # Repeat previous token
                    tokens.append(tokens[-1] if tokens else random.randint(0, 50256))
                else:
                    tokens.append(random.randint(0, 50256))
            gen_tokens.append(tokens)
        
        snapshot = TrainingSnapshot(
            step=step,
            timestamp=time.time(),
            policy_logprobs=policy_lp,
            reference_logprobs=ref_lp,
            reward_scores=rewards,
            generated_tokens=gen_tokens,
            loss=base_loss - step * 0.003 + random.gauss(0, 0.1),
            learning_rate=lr,
            gradient_norm=random.gauss(1.0, 0.3),
        )
        
        # Evaluate
        report = evaluator.evaluate(snapshot)
        
        # Handle interventions
        if report.intervention == InterventionType.CHECKPOINT_REVERT:
            logger.info(f">>> EXECUTING CHECKPOINT REVERT at step {step}")
            evaluator.execute_revert()
            lr *= 0.5  # Also reduce LR after revert
            base_kl = 2.0  # Reset simulation
        elif report.intervention == InterventionType.REDUCE_LR:
            lr *= evaluator.lr_reduction_factor
            logger.info(f">>> REDUCING LR to {lr:.2e} at step {step}")
        
        # Periodic status
        if step % 100 == 0:
            print(f"\n--- Step {step} ---")
            print(f"  KL: {report.kl_divergence:.4f} (trend: {report.kl_trend})")
            print(f"  Hack Score: {report.reward_hack_score:.3f}")
            print(f"  Repetition: {report.repetition_ratio:.3f}")
            print(f"  Entropy: {report.entropy:.3f}")
            print(f"  Intervention: {report.intervention.value}")
    
    # Final summary
    summary = evaluator.get_summary()
    print(f"\n{'=' * 70}")
    print("TRAINING RUN SUMMARY")
    print(f"{'=' * 70}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n--- Prometheus Metrics ---")
    print(evaluator.get_prometheus_metrics())
    
    return evaluator


if __name__ == "__main__":
    evaluator = simulate_training_run()
