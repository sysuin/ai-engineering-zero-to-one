# Reinforcement fine-tuning optimises what the grader rewards. A model with three ways of
# answering, a grader with a loophole, and the drift penalty of 08_preference_beta.py.
# Every number below is an assumption stated here; the point is the shape. No model calls.

import json

import numpy as np

STYLES = ["works the problem", "answers quickly", "formats a guess"]
TRUE = np.array([0.70, 0.50, 0.20])        # how often each style is actually right
# The loophole: the grader checks the answer's format and, for a well-formatted wrong answer,
# fails to notice it is wrong 85% of the time. Only the third style formats every answer.
LEAKY = TRUE + np.array([0.0, 0.0, 0.85]) * (1 - TRUE)
FIXED = TRUE.copy()                        # a grader that checks the value itself
REFERENCE = np.array([0.50, 0.40, 0.10])   # how the model answers before training


def train(reward: np.ndarray, drift_penalty: float, steps: int, rate: float = 0.5):
    """Exact policy-gradient ascent on E[reward] - penalty * KL(policy || reference)."""
    logits = np.log(REFERENCE)
    history = {}
    for step in range(steps + 1):
        policy = np.exp(logits) / np.exp(logits).sum()
        if step in (0, 100, steps):
            history[step] = policy
        kl = float(np.sum(policy * np.log(policy / REFERENCE)))
        advantage = reward - policy @ reward
        drift = np.log(policy / REFERENCE) - kl
        logits += rate * policy * (advantage - drift_penalty * drift)
    return history


print(f"  {'grader':<8}{'penalty':>8}{'step':>6}{'P(formats a guess)':>20}"
      f"{'grader score':>14}{'truly right':>13}")
for grader_name, reward in (("leaky", LEAKY), ("fixed", FIXED)):
    for penalty in ((0.0, 0.1, 0.5) if grader_name == "leaky" else (0.0,)):
        for step, policy in train(reward, penalty, steps=2_000).items():
            print(f"  {grader_name:<8}{penalty:>8}{step:>6}{policy[2]:>20.0%}"
                  f"{policy @ reward:>14.0%}{policy @ TRUE:>13.0%}")
        print()

# Where the penalty leaves the model in the end: the reference reweighted by exp(reward / penalty).
for penalty in (0.1, 0.5):
    tilt = REFERENCE * np.exp(LEAKY / penalty)
    print(f"  leaky grader, penalty {penalty}: the limit puts "
          f"{(tilt / tilt.sum())[2]:.0%} on formatting a guess")


def curve(reward: np.ndarray, drift_penalty: float, steps: int = 2_000, rate: float = 0.5):
    """The same ascent as train(), recording the grader's score and the truth at every 50th step."""
    logits, points = np.log(REFERENCE), []
    for step in range(steps + 1):
        policy = np.exp(logits) / np.exp(logits).sum()
        if step % 50 == 0:
            points.append([step, float(policy @ reward), float(policy @ TRUE)])
        kl = float(np.sum(policy * np.log(policy / REFERENCE)))
        drift = np.log(policy / REFERENCE) - kl
        logits += rate * policy * ((reward - policy @ reward) - drift_penalty * drift)
    return points


json.dump({"leaky, no penalty": curve(LEAKY, 0.0), "leaky, penalty 0.5": curve(LEAKY, 0.5),
           "fixed grader": curve(FIXED, 0.0)}, open("code/30/_grader_loophole.json", "w"))
