# CCR.GB: Compositional Causal Reasoning Evaluation in Language Models (paper reproducing repository)

This repository contains the codebase for the paper *Compositional Causal Reasoning Evaluation in Language Models* ([ICML 2025](https://arxiv.org/abs/2503.04556)). The CCR.GB benchmark is designed to evaluate LLMs on compositional causal reasoning across all three levels of Pearl's Causal Hierarchy: (1) association, (2) intervention, and (3) counterfactuals.

For a comprehensive overview of the research, visit the [main project page](https://jmaasch.github.io/ccr/).

Group's members information:

* 25C11039 - To Tan Hiep

* 25C11028 - Nguyen Hoang Viet

---

## Repository Walkthrough & Notebooks

We have implemented two main notebooks in the root of the project to demonstrate, verify, and reproduce the paper's findings:

1.  **[experimental_results.ipynb]:** 
    *   **Objective:** Reproduces and visualizes the key experimental results from **Section 6** of the paper.
    *   **Features:** Dynamically computes the ground truth values and simulates LLM response profiles (for models such as `o1`, `GPT-4o + CoT`, and `Llama 3`) based on the empirical statistics reported in the study.
    *   **Reproduced Outputs:** Recreates the Validity vs. Consistency scatter plot (**Figure 10**), the Commutative Cut Tree (CCT) reasoning profiles (**Figure 11**), and the error scaling over path length (**Figure 12**).
2.  **[verification.ipynb]:**
    *   **Objective:** Executes the step-by-step structural verification of the CCR evaluation pipeline.
    *   **Features:** Constructs the causal DAG, generates prompt contexts (factual and counterfactual), calculates the exact ground truth PNS values via Structural Causal Model (SCM) simulation ($n = 100,000$), and performs the verification of Theorem 5.1.

---

## Detailed Analysis of `verification.ipynb` Results

Below is a rigorous analysis of the execution results from [verification.ipynb] under the seed configuration (`np.random.seed(0)` and `random.seed(0)`).

### 1. Causal Graph & Node Identification
For a three-BCC graph configured with `n_per_bcc = [4, 3, 3]` and cycle types, the task generator constructs a Directed Acyclic Graph (DAG) with **8 nodes** and **2 cutpoints** (articulation points). The dynamically assigned node labels (randomly drawn from female names via `Faker`) are:
*   **Root ($X$):** `Nereida`
*   **Cutpoint 1 ($C$):** `Julia`
*   **Cutpoint 2 ($D$):** `Celie`
*   **Leaf ($Y$):** `Tristan`

The resulting Commutative Cut Tree (CCT) path sequence is:
$$X (\text{Nereida}) \to C (\text{Julia}) \to D (\text{Celie}) \to Y (\text{Tristan})$$

### 2. Ground Truth PNS Values
Using $100,000$ SCM simulation samples, we estimated the Probability of Necessity and Sufficiency (PNS) for all cause-effect pairs in the CCT:
*   **Global PNS ($PNS_{XY}$):** $0.000390$
*   **Local PNS ($PNS_{XC}$):** $0.048060$
*   **Local PNS ($PNS_{CY}$):** $0.009670$
*   **Local PNS ($PNS_{XD}$):** $0.005730$
*   **Local PNS ($PNS_{DY}$):** $0.081460$
*   **Local PNS ($PNS_{CD}$):** $0.121130$

**Key Observation:** The global PNS ($0.000390$) is exceptionally small. Because all causal functions are logical `or` and nodes are connected in series, the counterfactual probability of the leaf changing state in response to the root decays rapidly across multiple layers of mediating variables. The local effects are substantially larger (e.g., $PNS_{CD} \approx 12\%$), demonstrating that causal influence is highly localized.

### 3. Theorem 5.1 Verification & Monte Carlo Error
Theorem 5.1 states that for serial cutpoint structures, the global PNS equals the product of local PNS values along any CCT path. The notebook verified this theorem across three compositions:
*   **Composition 1 ($PNS_{XC} \times PNS_{CY}$):** $0.000465$ (Relative Absolute Error, **RAE: 19.16%**)
*   **Composition 2 ($PNS_{XD} \times PNS_{DY}$):** $0.000467$ (Relative Absolute Error, **RAE: 19.68%**)
*   **Composition 3 ($PNS_{XC} \times PNS_{CD} \times PNS_{DY}$):** $0.000474$ (Relative Absolute Error, **RAE: 21.59%**)

#### **Why do all three compositions deviate from the global PNS by 19%–22%?**
1.  **Finite-Sample Sampling Variance:** This deviation is a finite-sample Monte Carlo sampling artifact, not a violation of Theorem 5.1.
2.  **Small Global Probability Sensitivity:** The true global probability $PNS_{XY} = 0.000390$ translates to observing only $\approx 39$ joint events out of $100,000$ samples. The expected standard error of this estimation is $\sqrt{39}/100,000 \approx 0.000062$, giving a high coefficient of variation (relative standard error) of $\approx 16\%$.
3.  **Cumulative Product Variance:** Compositions multiply two or three independent estimates. Since each local PNS has its own sampling variance, multiplying them compounds their relative error.
4.  **Directional Bias:** At finite $n$, multiplying positive, independently perturbed small probabilities leads to a systematic upward bias, explaining why all three compositions overestimate the true global PNS.

### 4. Cross-Theme Structural Consistency
*   **Theme Equivalence:** The notebook verified that `FluVaccine` and `FlowerGarden` DAGs match `CandyParty` structurally (`True`). This confirms that the benchmark isolates reasoning ability from semantic context: the causal graphs, SCMs, and math-logical parent-child relationships remain identical; only the verbal surface form changes.
*   **ClinicalNotes Variant:** `ClinicalNotes` enforces logical `or` for all intermediate relations but fixes the final leaf-node function to `and` (representing that a recommendation for surgery requires all diagnostic conditions to be met).
*   **CellBio Variant:** `CellBio` targets the Average Treatment Effect (ATE) instead of the PNS. It uses a fundamentally different linear SCM process with continuous variables and Gaussian noise, representing gene transcription volumes.

---

## Setup & Running the Notebooks

Ensure that you have installed the required dependencies in the virtual environment.

### 1. Activate the Environment
```powershell
# On Windows PowerShell:
.venv\Scripts\activate
```
or activate your own enviroment if using others.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Make sure `pandas`, `numpy`, `networkx`, and `matplotlib` are installed.)*

### 3. Launch Jupyter
```bash
jupyter notebook
```
Open **[experimental_results.ipynb]** or **[verification.ipynb]** to inspect the results. Since the notebooks are pre-executed, you can view the rendered plots and tables immediately upon opening.
