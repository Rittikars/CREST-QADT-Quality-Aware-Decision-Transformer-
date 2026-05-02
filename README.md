# Quality Aware Decision Transformer for Offline Reinforcement Learning

A university project extending **Decision Transformer: Reinforcement Learning via Sequence Modeling** with a practical quality aware training pipeline for offline Atari reinforcement learning.

## Project Overview

This project starts from the original Decision Transformer idea of treating reinforcement learning as a sequence modeling problem. Instead of using value functions and Bellman updates, the model predicts the next action from a sequence of return to go, state, and action tokens.

Our final project extends that idea with a **Quality Aware Decision Transformer**, where trajectory quality is explicitly used during training. The goal was not only to reproduce the paper, but to add a technically meaningful improvement and evaluate whether quality based weighting can help offline learning on Atari games.

## Main Contribution

The key contribution of this project is a **quality aware training mechanism** added on top of the Decision Transformer framework.

### What we added beyond the paper

- **Confidence guided training loss**
  - Each episode is assigned a confidence score based on trajectory quality.
  - During training, the action prediction loss is weighted by this confidence.
  - Higher quality trajectories influence optimization more strongly.

- **Rank normalized confidence scoring**
  - Instead of relying only on raw return scaling, episodes can be ranked by return.
  - This makes the method more robust to outlier trajectories.

- **Timestep decaying confidence**
  - Confidence can decay across the episode.
  - This allows earlier actions to receive relatively higher importance when desired.

- **Confidence blend ablation**
  - The project includes experiments over different confidence blend values.
  - This helps analyze how much quality weighting is actually useful.

- **Noise robustness experiment**
  - Plain DT and Quality Aware DT can be compared under noisy offline data.
  - This makes the work more research oriented than a simple reproduction.

- **Integrated result pipeline**
  - The code automatically saves learning curves, comparison tables, summary files, and gameplay videos.

## Implemented Environments

The main experiments were conducted on Atari offline RL datasets using Minari.

### Atari games used

- Qbert
- Pong
- Breakout
- Seaquest
- BeamRider

## Model Design

### Baseline Plain DT

For each timestep, the model processes:

- Return to go
- State
- Action

The model uses:

- CNN state encoder for image observations
- Transformer encoder with causal mask
- Cross entropy loss for discrete action prediction

### Quality Aware DT

The proposed model adds one more token:

- **Confidence token**

So each timestep becomes:

- Return to go
- Confidence
- State
- Action

The loss becomes:

`weighted loss = cross entropy × confidence weight × mask`

where the confidence weight is controlled by the **confidence blend** parameter.

## Technical Pipeline

1. Load offline Atari trajectories from Minari.
2. Downsample episode frames for stable and cheaper training.
3. Compute per episode return and return to go.
4. Compute confidence from episode quality.
5. Convert episodes into sliding window training samples.
6. Train both Plain DT and Quality Aware DT.
7. Evaluate each model by rollout.
8. Save metrics, plots, tables, and gameplay videos.

## Files in the Project

### Main code files

- `confidence_guided_dt_atari.py`
  - Baseline comparison code for Plain DT and Quality Aware DT.

- `quality_dt_enhanced.py`
  - Enhanced version with rank normalized confidence, timestep decay, ablation, and noise experiments.

### Output artifacts

Typical output folders contain:

- `config.json`
- `summary.json`
- `summary.txt`
- `comparison_metrics.csv`
- `comparison_table.png`
- `paper_style_bc_table.png`
- `return_plot.png`
- `learning_curves.png`
- `dataset_episode_video.mp4`
- `model_rollout_video.mp4`

## Core Hyperparameters Used

Typical project settings included:

- Context length: 4 to 30 depending on the run
- Epochs: 10 to 20
- Confidence mode: `linear` or `rank`
- Confidence blend: commonly `0.2` or `0.6`
- Device: CPU in local runs

## Final Experimental Findings

The final project does **not** claim that Quality Aware DT wins on every game.

Instead, the results show a more realistic conclusion:

- Quality aware training improved performance on several games.
- The effect depended on the game and the confidence design.
- Rank based confidence with lower blend was often more stable than naive weighting.
- Some games such as Seaquest, Breakout, and BeamRider showed clearer gains.
- Qbert required tuning before Quality Aware DT became competitive.
- Pong showed only modest improvement.

This makes the project stronger academically because it presents a real experimental conclusion rather than forcing a universal success claim.

## Why this project is different from a direct paper reproduction

This project is not only a reimplementation of Decision Transformer.

It adds:

- quality aware token based conditioning
- weighted training objective
- confidence ranking mechanism
- timestep decayed weighting
- ablation study support
- noise robustness testing
- automated visual result generation

These additions make the work more suitable as a university final project because they show both implementation skill and experimental thinking.

## Practical Future Work

- Test the method on more Atari games.
- Tune confidence parameters separately for each game.
- Run more seeds for stronger statistical comparison.
- Compare directly with behavior cloning and percentile BC.
- Increase evaluation coverage with more rollout episodes.
- Extend the same method to continuous control tasks such as HalfCheetah.

## How to Run

### Example standard run

```bash
python quality_dt_enhanced.py --game qbert --epochs 10 --episode_limit 10
```

### Example rank confidence run

```bash
python quality_dt_enhanced.py --game qbert --epochs 20 --episode_limit 50 --confidence_mode rank --confidence_blend 0.2
```

### Example ablation sweep

```bash
python quality_dt_enhanced.py --game qbert --epochs 10 --episode_limit 10 --run_ablation
```

### Example noise experiment

```bash
python quality_dt_enhanced.py --game qbert --epochs 10 --episode_limit 10 --run_noise
```

## Requirements

Typical Python packages used in this project:

- Python
- PyTorch
- NumPy
- Matplotlib
- Minari
- Gymnasium or Gym
- OpenCV
- CSV and JSON utilities

## Presentation Focus

For the final presentation, the most important message of this project is:

> We extended Decision Transformer with a quality aware learning mechanism and showed that trajectory quality based weighting can improve offline RL performance on multiple Atari games, while also providing a full evaluation and visualization pipeline.

## Acknowledgment

This project is based on the paper:

**Decision Transformer: Reinforcement Learning via Sequence Modeling**

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch

---

If you use this repository, please cite the original Decision Transformer paper and clearly distinguish the original method from the project specific quality aware extensions.
