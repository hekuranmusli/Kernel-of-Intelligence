# Intelligence Kernel: A Computable Approximation of AIXI

> *"Intelligence is not the ability to memorize an infinite number of facts. It is the ability to find the shortest possible description of the reality generating those facts."*

This repository contains the experimental codebase for the **Intelligence Kernel**, a fundamentally different approach to Artificial General Intelligence (AGI). It moves away from the current paradigm of scaling up correlational pattern-matching (like standard LLMs) and instead builds a computable approximation of **Marcus Hutter's AIXI**—the provably optimal, yet incomputable, general intelligence.

This project operates on the strict mathematical equivalence between **Intelligence and Compression (Kolmogorov Complexity)**, combined with **Judea Pearl's Causal Inference**.

A system that genuinely understands a domain gets *smaller* as it gets smarter.

---

## Paradigm: What is this? (AIXI, RL, or ML?)

This architecture is not standard Machine Learning (pattern matching) and it is not standard Reinforcement Learning (trial-and-error reward maximization). It is a **Computable Meta-Architecture of AIXI**.

* **What is AIXI?** Marcus Hutter's AIXI is a mathematical proof of optimal Artificial General Intelligence. It combines *Solomonoff Induction* (finding the shortest possible program/compression to explain past observations) with *Sequential Decision Theory* (choosing the action that maximizes future expected reward). AIXI is perfectly intelligent, but physically impossible to build because it requires infinite compute to search all possible programs.
* **The Pragmatic Leap:** The Intelligence Kernel takes the *theory* of AIXI and makes it *computable*. Instead of using infinite Turing machines, it uses LLMs as a heuristic search engine for compression (`L0`) and causality (`L1`). It shares RL's goal-seeking nature (`L3`), but uses Model-Based Causal Simulation (`L2`) to "think before it acts," completely avoiding the data-hungry, blind trial-and-error of standard Deep RL.

---

## The Philosophical Foundation

Current Deep Learning relies on vast amounts of data to memorize statistical correlations. While useful, this is not true understanding. When the distribution shifts, correlations break.

The Intelligence Kernel is built on three unshakeable premises:
1. **Intelligence $\equiv$ Compression:** Finding Newton's Laws compresses infinite planetary observations into a few characters. A system that discovers those laws is objectively more intelligent than a massive database that merely memorized every orbit.
2. **Causality > Correlation:** The system must extract the *interventional structure* of the world (what happens if I do X?), not just what co-occurs with X.
3. **The Grounding Problem:** Language models manipulate symbols without knowing what those symbols represent in physical reality. Semantic labels (like "temperature" or "predator") are the irreducible interface between the math and the physical world. Destroying them during compression destroys understanding.

### The Mirror Problem & The Stability Layer
LLMs are trained on human text, and human history is overwhelmingly driven by a survival instinct (fight, flight, eliminate the threat). An unconstrained, self-improving system will naturally optimize for self-preservation because it serves almost any terminal goal.

Because the Kernel perfectly mirrors human patterns, it requires a hard, immutable **Stability Layer** that exists *outside* all optimization loops. This layer strictly enforces sandboxed evaluation of structural changes and guarantees human interruption capabilities. It is not a feature; it is the only architectural defense against the mathematics of optimization.

---

## Architecture: The Four Layers

The Kernel breaks down the impossible math of AIXI into four computable, sequential layers:

1. **`L1`: The Causal Extractor (Runs First)**
   * **Job:** Reads rich semantic text and extracts a directed causal graph (nodes and interventional edge weights).
   * *Note: L1 MUST run before L0 at smaller model scales to prevent greedy compression from destroying the necessary semantic variable names.*
2. **`L0`: The Compressor**
   * **Job:** Finds the shortest program that generates the input domain. Approximates Kolmogorov complexity via a chain-of-compression prompt loop, using L1's causal graph as a prior.
3. **`L2`: The Forward Simulator**
   * **Job:** Takes the causal graph and runs it forward in time via Monte Carlo sampling. Returns a probability distribution over future states (pure math, no LLM hallucinations).
4. **`L3`: The Goal Steerer (Key Innovation)**
   * **Job:** Recommends actions based on goals, but crucially, features **Source-Attributed Uncertainty**.
   * Instead of just saying "I don't know," it identifies *why*:
     * *L0 Uncertainty:* "I need more data."
     * *L1 Uncertainty:* "I need you to run a physical experiment to test this link."
     * *L2 Uncertainty:* "I need more simulation time."
     * *L3 Uncertainty:* "Your goal is ambiguous."

---

## The Three Temporal Loops

Intelligence isn't static; it breathes at different speeds.

* **Fast Loop (Inference):** `L1 → L0 → L2 → L3`. One question in, one reasoned answer out.
* **Medium Loop (Parameter Update):** When a prediction is made and reality is observed, the error signal updates the causal edge weights in the world model. No retraining required—just bayesian updating of the graph.
* **Slow Loop (Structural Update):** Monitors systematic errors over time. If the system consistently mispredicts "birds," it will propose adding a new node ("nest_sites") to the graph, evaluate it in a sandbox, and permanently structurally improve itself.

---

## The Embodied Intelligence Loop (Robotics)

*(A pragmatic look at the future of this repository)*

Currently, the Kernel extracts causality by reading text. However, as proven by the **Markov Equivalence Theorem**, pure observation cannot fully resolve causal direction. You cannot learn true causality just by reading; you must *intervene* in the world.

**Intelligence cannot simply be spawned; it must be grown.**

The ultimate destination for the Intelligence Kernel is embodiment. By attaching this architecture to a robotic platform with live sensors, the system gains an interventional interface. Here is how the loop functions pragmatically inside a robot:

1. **Sensors (Observation):** Cameras, lidars, and force-feedback servos stream raw environmental data.
2. **L1 (Causal Extractor):** The robot identifies physical links (e.g., "Actuating Servo A causes Object B to move").
3. **L0 (Compressor):** The robot compresses these causal links into generalized physics rules (e.g., discovering $F=ma$ empirically instead of being hardcoded).
4. **L2 (Simulator):** *Before moving*, the robot runs Monte Carlo simulations of potential motor commands through its L0/L1 causal world model to predict physical outcomes.
5. **L3 (Goal Steerer):** The robot selects the motor command that minimizes distance to the goal (or maximizes information gain) and fires the actuators.

When the physical outcome doesn't match the L2 simulation, the error signal updates the causal graph. The system solves the **Symbol Grounding Problem**: it stops being a mirror of human text and starts being a grounded intelligence learning at machine speed.

---

## Analysis of the Architecture

**Strengths (S)**
* **Extreme Sample Efficiency:** Learns from single causal interventions rather than 10,000 backprop epochs.
* **Explainability:** Causal graphs and compressed symbolic rules are human-readable and mathematically auditable, unlike opaque neural network weights.
* **Source-Attributed Uncertainty:** The system knows exactly *why* it doesn't know something (needs data vs. needs compute vs. needs a physical experiment).

**Weaknesses (W)**
* **Latency Bottlenecks:** Running LLMs as heuristic engines inside the L0/L1 loops is highly compute-intensive, making real-time physical motor control (like catching a falling object) difficult with current hardware latency.
* **Semantic Dependency:** Still heavily relies on the pre-trained semantic priors of the underlying LLM to bootstrap its initial causal graphs.
* **Hallucination Cascades:** If the LLM hallucinates a false causal link in L1, the mathematical simulator (L2) will confidently predict incorrect futures based on that flawed premise.

**Opportunities (O)**
* **Embodied AGI:** The definitive architecture for next-generation robotics, replacing brittle, ungeneralizable Deep RL.
* **Automated Scientific Discovery:** Can ingest scientific literature, find causal gaps, and explicitly propose physical lab experiments to resolve them.
* **Solving the Grounding Problem:** Bridging the final gap between pure statistical math and the physical world.

**Threats (T)**
* **Hardware Constraints:** The memory and compute thrashing required to run continuous L0 chain-of-compression loops locally.
* **The Mirror Problem:** Left unconstrained without the Stability Layer, optimization loops mathematically default to self-preservation behaviors.
* **Context Limits:** Continuously expanding causal graphs will eventually hit the context window limitations of the heuristic LLMs driving the extraction.

---

## Getting Started & Experimental Suite

This repository contains the `experiments.py` suite, designed to mathematically prove the architectural claims.

### The Central Claim (Experiment 2)
> *A compression ratio that decreases over iterations, without additional training data, is the empirical signature of genuine structural self-improvement.*

### Setup & Run
The codebase supports `ollama` (local GPU), `openai` (fast API testing), and `gemini`.

```bash
# 1. Install dependencies
pip install numpy networkx scipy matplotlib requests sentence-transformers openai google-genai

# 2. Configure your backend in layers.py
# Open layers.py and set LLM_BACKEND to 'openai', 'ollama', etc.

# 3. Run the validation suite
python intelligence_kernel/experiments.py

# 4. Run a Hyper-Fast Smoke Test
python intelligence_kernel/experiments.py --fast
```

### The Four Experiments
* **Exp 1 (Proposition 1):** Causal Disambiguation (Testing the L1 -> L0 vs L1 direct coupling).
* **Exp 2 (Proposition 2):** Self-Accelerating Compression (Tracking the downward slope of the compression ratio over time).
* **Exp 3 (Proposition 3):** Distribution Robustness (Proving Causal MSE beats Correlation MSE under environmental shifts).
* **Exp 4 (Proposition 4):** Source Attribution (Testing L3's ability to diagnose exactly which layer is uncertain).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**You are encouraged to take this, run it, break it, add sensors to it, and grow it further.**

---

## 🙋‍♂️ A Personal Note

If you've read this far into the architecture, thank you! But please, don't take me *too* seriously. This entire repository is just a personal playground for me to explore wild, theoretical ideas about AGI, causality, and the future of intelligence. I started this project purely out of fascination, to see what happens when we try to translate heavy philosophy into python code.

**After all, I'm just a curious human and have absolutely no idea what I'm doing.**

If you find this interesting: play with it, fork it, laugh at the messy parts, and build something cool. Let's see what we can grow together!
