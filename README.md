# Shadow Evaluator for RL Training Stability

> Real-time sidecar monitor that catches mode collapse, reward hacking, and policy drift in RLHF training runs before they waste thousands of dollars in compute.

## The Problem

RLHF training runs are expensive and fragile. A single mode collapse or reward hacking event can waste days of GPU time and thousands of dollars. Current approaches rely on post-hoc evaluation by the time you notice the problem, the damage is done.

## The Solution

A deterministic fail-safe system that monitors RL training in real-time:

```
Training Loop                    Shadow Evaluator (Sidecar)
─────────────                    ─────────────────────────
                                 
  Policy Model ──┐               ┌──→ KL Divergence Monitor
                 │               │     • Real-time KL(π || π_ref)
  Reference   ───┤  Snapshot     │     • Trend detection
  Model          ├──────────────►├──→ Reward Hacking Detector
                 │   every N     │     • Repetition analysis
  Reward      ───┤   steps       │     • Entropy monitoring
  Model          │               │     • Reward spike detection
                 │               │
                 │               ├──→ Intervention Engine
                 │               │     • Auto checkpoint revert
                 │               │     • Learning rate reduction
                 │               │     • Hard stop
                 │               │
                 │    Revert     ├──→ Checkpoint Manager
                 │◄──────────────┤     • Health-tagged checkpoints
                 │               │     • Instant rollback
                 │               │
                 │               └──→ Prometheus Metrics
                                       • rl_kl_divergence
                                       • rl_reward_hack_score
                                       • rl_interventions_total
```

## Detection Capabilities

| Failure Mode | Detection Method | Response |
|---|---|---|
| **Mode Collapse** | KL divergence spike + entropy drop | Checkpoint revert + LR reduction |
| **Reward Hacking** | Token repetition + reward spike | Immediate checkpoint revert |
| **Policy Drift** | KL trend analysis (rising) | LR reduction after N warnings |
| **Training Instability** | Gradient norm + loss monitoring | Alert + potential hard stop |

## Usage

```python
from shadow_evaluator import ShadowEvaluator, TrainingSnapshot

evaluator = ShadowEvaluator(
    kl_warning=10.0,
    kl_critical=25.0,
    checkpoint_interval=100,
)

for step in range(num_steps):
    # Your RLHF training code...
    
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
```

## Running the Demo

```bash
python shadow_evaluator.py
```

Simulates a 500-step RLHF training run with injected failure modes:
- Steps 0-150: Healthy training
- Steps 150-250: KL drift (early warning)
- Steps 250-350: Reward hacking begins
- Steps 350-500: Mode collapse (caught and reverted)

## Prometheus Integration

```python
# Expose metrics endpoint
metrics = evaluator.get_prometheus_metrics()
# Returns Prometheus exposition format
```

## Production Roadmap

- [ ] PyTorch hook integration for activation monitoring
- [ ] Grafana dashboard templates
- [ ] Multi-GPU distributed monitoring
- [ ] Slack/PagerDuty alerting integration
- [ ] Historical run comparison database

## Author

Uday — ML Infrastructure Engineer | Ex-Google DeepMind (Gemini)
