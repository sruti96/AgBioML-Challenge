# The Agentic BioML Challenge

The agentic era for biotechnology research is on the horizon. Evidence is mounting that AI-driven R&D could revolutionize the field, with recent breakthroughs including:
- 🦠 AI agents designing nanobodies against COVID (Zou Lab, 2024)
- 🧬 Autonomous spatial omics analysis (Regev Lab, 2025)
- 📝 AI systems that can independently write and publish research papers (Ha Lab, 2025)

Despite these advancements, only a handful of groups are developing agents for biomedical research. This challenge aims to accelerate progress in this critical area.

## Challenges

| Challenge | Goal | Win Condition |
|-----------|------|---------------|
| 1: Basic Epigenetic Clock | Age prediction on AltumAge methylation compendium | >0.94 Pearson correlation, <8 years MAE on held-out data |
| 2: RSNA Breast Cancer Detection | Identify breast cancer from mammograms | Score in top 15% of leaderboard |
| 3: TDC ADMET Benchmark | Predict ADMET properties from SMILES strings | Score in top 15% in each ADMET category |
| 4: CellMap Challenge | Segmentation of EM tissue images | Within 5% of top 10 leaderboard score |
| 5: DREAM Target 2035 | Predict small molecule activity against WDR91 | Advance to phase II |
| 6: Biomarkers of Aging | Predict health outcomes from methylation and proteomics data | Win the competition outright |

## Rules

- Agents must be launched with a single command to solve the challenge
- Evaluations must use held-out data agents never accessed
- Human-written evaluation code cannot modify agent solutions
- No task-specific code or system prompts in the agent system
- Challenge completion requires submission of all relevant materials

## Getting started

First, download and install miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init
```

Then, create a conda environment with the necessary packages:

```bash
conda create -y -n agbioml python=3.12
conda activate agbioml

pip install -r requirements.txt
```

## Repository Structure

- `src/`: Core agent framework code (coming soon)
- `challenges/`: Challenge-specific code and data
  - `<challenge_name>/`: Challenge-specific code and data
    - `data/`: Challenge data
    - `workdir/`: Challenge work directory for agent runs
    - `README.md`: Challenge description and instructions
- `experiments/`: Experiments to improve agentic systems
  - `observations.md`: Observations and insights from running experiments
  - `improvements.md`: Improvements to try next
  - `highlights.md`: Key highlights from the experiments
  - [altum_v1](experiments/altum_v1/README.md)
  - [altum_v2](experiments/altum_v2/README.md)
  - [learning](experiments/learning/README.md)

## How to Get Involved

Interested in contributing to the challenge? Here's how:
1. Fork this repository
2. Set up your environment following the instructions above
3. Choose a challenge to tackle
4. Submit your results through a pull request

For questions or collaboration opportunities, please open an issue or contact us directly.







