"""
Intelligence Kernel — Core Layer Implementations
L0: Compressor  |  L1: Causal Extractor  |  L2: Simulator  |  L3: Steerer

LLM_BACKEND: set to 'ollama' (real) or 'ollama' (testing/demo).
On your machine with Mistral: set LLM_BACKEND = 'ollama'
"""

import json
import re
import time
import random
import numpy as np
import networkx as nx
from typing import Optional
from dataclasses import dataclass, field
from collections import deque

# ─── Configuration ────────────────────────────────────────────────
LLM_BACKEND   = 'openai'   # 'ollama', 'gemini', 'openai', or 'mock'
LLM_MODEL     = 'gpt-4o-mini'
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY_HERE'
UNCERTAINTY_THRESHOLD = 0.30
SLOW_LOOP_INTERVAL    = 20   # iterations between slow-loop checks

# ─── LLM Interface ────────────────────────────────────────────────
def llm_call(prompt: str, retries: int = 3) -> str:
    """Route to real Ollama, Gemini, OpenAI, or mock depending on LLM_BACKEND, with retries."""

    # === ADDED FOR VISIBILITY ===
    # print(f"\n[{time.strftime('%H:%M:%S')}] > SENDING PROMPT TO {LLM_BACKEND.upper()} ({LLM_MODEL}):")
    # print("-" * 40)
    # print(prompt.strip())
    # print("-" * 40)

    for attempt in range(retries):
        try:
            if LLM_BACKEND == 'ollama':
                import ollama
                r = ollama.chat(model=LLM_MODEL,
                                messages=[{'role': 'user', 'content': prompt}])
                response_text = r['message']['content']
            elif LLM_BACKEND == 'gemini':
                from google import genai
                client = genai.Client(api_key=GEMINI_API_KEY)
                response = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt,
                )
                response_text = response.text
            elif LLM_BACKEND == 'openai':
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content
            else:
                response_text = _ollama_llm(prompt)

            # === ADDED FOR VISIBILITY ===
            # print(f"[{time.strftime('%H:%M:%S')}] > RECEIVED RESPONSE:")
            # print("~" * 40)
            # print(response_text.strip())
            # print("~" * 40 + "\n")

            return response_text

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ! ERROR (Attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"LLM call failed after {retries} attempts: {e}")
                return "{}" # Return empty JSON-like string on total failure

def _extract_json_from_text(text: str) -> dict:
    """Robustly extract JSON object from LLM text output."""
    try:
        # First, try to find a JSON code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # If no code block, try to find the outermost curly braces
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Fallback to empty default
        return {"edges": [], "uncertainty": 0.9}
    except json.JSONDecodeError:
        return {"edges": [], "uncertainty": 0.9}

def _ollama_llm(prompt: str) -> str:
    """
    Deterministic ollama that returns plausible-looking outputs.
    Gives the experiments real structure to work with.
    """
    prompt_lower = prompt.lower()

    # L0: compression request
    if 'simplest rule' in prompt_lower or 'executable python' in prompt_lower:
        seeds = [
            "def gen(n): return [i**2 for i in range(n)]",
            "def gen(x): return x * 1.618 + 0.5",
            "def gen(s): return sorted(set(s.split()))",
            "def gen(n): return [2**i for i in range(n)]",
            "def gen(x,a=2,b=1): return a*x + b",
        ]
        base = random.choice(seeds)
        # each compression pass shortens it slightly
        if 'more concisely' in prompt_lower:
            base = base.replace('def gen', 'f=lambda')
            base = base.replace(': return', ':')
        if 'can this program be shortened' in prompt_lower:
            base = re.sub(r'\s+', ' ', base)
        return base

    # L1: causal graph extraction
    if 'causal relationship' in prompt_lower or 'control flow' in prompt_lower:
        nodes = ['input', 'process', 'state', 'output', 'feedback']
        n_edges = random.randint(3, 6)
        edges = []
        used = set()
        for _ in range(n_edges):
            i = random.randint(0, len(nodes)-2)
            j = random.randint(i+1, len(nodes)-1)
            key = (i, j)
            if key not in used:
                used.add(key)
                edges.append({
                    "cause":     nodes[i],
                    "effect":    nodes[j],
                    "strength":  round(random.uniform(0.4, 0.95), 2),
                    "mechanism": f"{nodes[i]} directly drives {nodes[j]}",
                    "testable":  True
                })
        uncertainty = round(random.uniform(0.15, 0.45), 2)
        return json.dumps({"edges": edges, "uncertainty": uncertainty})

    # L3: goal → target node
    if 'which node' in prompt_lower and 'goal' in prompt_lower:
        lines = prompt_lower.split('\n')
        for line in lines:
            if 'available nodes' in line:
                match = re.findall(r"'([^']+)'", line)
                if match:
                    return random.choice(match)
        return 'output'

    return "I processed that request."


# ─── Data Structures ──────────────────────────────────────────────
@dataclass
class CompressionResult:
    program:          str
    length:           int
    compression_ratio: float
    uncertainty:      float
    passes:           list
    iteration:        int = 0

@dataclass
class CausalResult:
    graph:       nx.DiGraph
    uncertainty: float
    n_nodes:     int
    n_edges:     int

@dataclass
class SimulationResult:
    distribution: dict   # node -> {mean, std, p95}
    uncertainty:  float  # mean std across all nodes

@dataclass
class SteerResult:
    action:           Optional[str]
    target_node:      Optional[str]
    confidence:       float
    uncertainty_source: str
    uncertainties:    dict
    diagnostic:       Optional[str] = None
    note:             Optional[str] = None

@dataclass
class WorldModel:
    """Persistent state that improves across iterations."""
    causal_graph:       nx.DiGraph = field(default_factory=nx.DiGraph)
    compression_prior:  dict       = field(default_factory=dict)
    error_history:      deque      = field(default_factory=lambda: deque(maxlen=1000))
    compression_log:    deque      = field(default_factory=lambda: deque(maxlen=1000))
    iteration:          int        = 0

    def record_compression(self, ratio: float):
        self.compression_log.append({'iteration': self.iteration,
                                     'ratio': ratio,
                                     'timestamp': time.time()})

    def record_error(self, predicted, actual, context=""):
        self.error_history.append({
            'iteration': self.iteration,
            'predicted': predicted,
            'actual':    actual,
            'context':   context,
            'timestamp': time.time()
        })


# ─── L0: Compressor ───────────────────────────────────────────────
class L0Compressor:
    """
    Chain-of-compression: iteratively asks LLM to shorten
    the generative description of input_text.
    Uses causal_prior from WorldModel to bias toward causally
    consistent programs (the L1->L0 feedback that drives acceleration).
    """

    INSTRUCTIONS = [
        "What is the simplest rule that generates this? CRITICAL: You MUST preserve all original semantic variable names (e.g. 'temperature', 'predator'). Do not use generic variables like x or y.",
        "Express that rule more concisely, but NEVER remove the semantic variable names.",
        "Write that rule as executable Python. CRITICAL: The variable names in the code MUST be the exact semantic words from the text, not math abstractions.",
        "Shorten this program, but you are FORBIDDEN from renaming any semantic variables to single letters. Keep the meaning intact.",
    ]

    def compress(self, input_text: str, world_model: WorldModel,
                 iteration: int = 0) -> CompressionResult:
        passes = []
        prompt = input_text

        # Causal prior: if we have a prior, prepend it as a hint
        prior_hint = ""
        if world_model.compression_prior:
            top = sorted(world_model.compression_prior.items(),
                         key=lambda x: x[1], reverse=True)[:3]
            prior_hint = f"Known causal primitives: {[k for k,v in top]}. "

        for i, instruction in enumerate(self.INSTRUCTIONS):
            full_prompt = f"{prior_hint}{instruction}\n\n{prompt}"
            result = llm_call(full_prompt)
            passes.append({'text': result, 'length': len(result), 'pass': i})
            prompt = result

        best = min(passes, key=lambda x: x['length'])

        # Compression uncertainty: how much did passes disagree?
        lengths = [p['length'] for p in passes]
        uncertainty = (max(lengths) - min(lengths)) / max(lengths) if max(lengths) > 0 else 0.0

        ratio = best['length'] / max(len(input_text), 1)
        cr = CompressionResult(
            program=best['text'],
            length=best['length'],
            compression_ratio=ratio,
            uncertainty=round(uncertainty, 3),
            passes=passes,
            iteration=iteration
        )
        world_model.record_compression(ratio)
        return cr


# ─── L1: Causal Extractor ─────────────────────────────────────────
class L1CausalExtractor:
    """
    Extracts causal graph from L0's compressed program.
    Control flow in a program is causal by construction — this is
    the key insight of Proposition 1.
    """

    def extract(self, compressed: CompressionResult,
                world_model: WorldModel) -> CausalResult:
        prompt = f"""
Analyse this program's control flow and extract causal relationships.
Program: {compressed.program}

Return JSON only:
{{"edges": [{{"cause": "X", "effect": "Y", "strength": 0.0-1.0,
             "mechanism": "brief description", "testable": true}}],
  "uncertainty": 0.0-1.0}}

Rules:
- Only include relationships derivable from control flow (if/then = causal)
- Do NOT include correlational relationships
- testable = can this be verified by intervention?
- uncertainty = your confidence in this graph overall
"""
        raw = llm_call(prompt)
        data = _extract_json_from_text(raw)

        def _node_str(val) -> str:
            """Flatten list/non-string node names Mistral sometimes returns."""
            if isinstance(val, list):
                return '_'.join(str(x) for x in val)
            return str(val).strip()

        G = nx.DiGraph()
        for e in data.get('edges', []):
            if e.get('testable', False):
                cause  = _node_str(e.get('cause',  ''))
                effect = _node_str(e.get('effect', ''))
                if cause and effect and cause != effect:
                    G.add_edge(cause, effect,
                               weight=e.get('strength', 0.5),
                               mechanism=e.get('mechanism', ''))

        # Merge with world model's existing graph (medium loop)
        for u, v, d in world_model.causal_graph.edges(data=True):
            if G.has_edge(u, v):
                # Blend: new extraction + existing knowledge
                G[u][v]['weight'] = 0.7 * G[u][v]['weight'] + 0.3 * d['weight']
            else:
                G.add_edge(u, v, **d)

        world_model.causal_graph = G

        # Update compression prior: nodes in causal graph
        # are good compression primitives (L1->L0 feedback)
        for node in G.nodes():
            world_model.compression_prior[node] = \
                world_model.compression_prior.get(node, 0) + 1

        return CausalResult(
            graph=G,
            uncertainty=float(data.get('uncertainty', 0.5)),
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges()
        )

    def update_from_error(self, world_model: WorldModel,
                          predicted_node: str, actual_outcome: str,
                          learning_rate: float = 0.1):
        """Medium loop: decay edges that contributed to wrong prediction."""
        G = world_model.causal_graph
        if predicted_node in G:
            for u in G.predecessors(predicted_node):
                if actual_outcome != predicted_node:
                    G[u][predicted_node]['weight'] *= (1.0 - learning_rate)
        world_model.record_error(predicted_node, actual_outcome)


# ─── L2: Forward Simulator ────────────────────────────────────────
class L2Simulator:
    """
    Monte Carlo simulation over causal graph.
    The graph is explicit causal structure — not learned latent dynamics.
    This is the source of Proposition 3 (distribution robustness).
    """

    def simulate(self, causal: CausalResult, initial_state: dict,
                 n_steps: int = 5, n_samples: int = 1000) -> SimulationResult:
        G = causal.graph
        nodes = list(G.nodes())

        if not nodes:
            return SimulationResult(distribution={}, uncertainty=1.0)

        results = []
        for _ in range(n_samples):
            state = {n: initial_state.get(n, np.random.uniform(0.1, 0.5))
                     for n in nodes}
            for _ in range(n_steps):
                new_state = state.copy()
                try:
                    order = list(nx.topological_sort(G))
                except nx.NetworkXUnfeasible:
                    order = nodes  # fallback if cycle
                for node in order:
                    parents = list(G.predecessors(node))
                    if not parents:
                        continue
                    influence = sum(
                        state.get(p, 0.0) * G[p][node].get('weight', 0.5)
                        for p in parents
                    )
                    noise = np.random.normal(0, 0.05)
                    new_state[node] = float(np.clip(influence + noise, 0.0, 1.0))
                state = new_state
            results.append(state)

        vals = np.array([[r.get(n, 0.0) for n in nodes] for r in results])
        dist = {}
        for i, node in enumerate(nodes):
            col = vals[:, i]
            dist[node] = {
                'mean': float(col.mean()),
                'std':  float(col.std()),
                'p95':  float(np.percentile(col, 95))
            }

        overall_uncertainty = float(np.mean([v['std'] for v in dist.values()]))
        return SimulationResult(distribution=dist, uncertainty=overall_uncertainty)


# ─── L3: Goal Steerer ─────────────────────────────────────────────
class L3Steerer:
    """
    Uncertainty-aware goal steering with SOURCE ATTRIBUTION.
    This is the novel contribution: not just 'I am uncertain'
    but 'I am uncertain because of [specific layer]' —
    producing qualitatively different responses per source.
    """

    DIAGNOSTICS = {
        'compression': "Uncertainty from L0: need more/better input data to find a shorter generative rule.",
        'causal':      "Uncertainty from L1: need an intervention to resolve causal ambiguity — run an experiment.",
        'simulation':  "Uncertainty from L2: need more simulation samples or longer horizon.",
        'goal':        "Uncertainty from L3: goal description is ambiguous — please clarify what you're optimising for."
    }

    def steer(self, goal: str, compressed: CompressionResult,
              causal: CausalResult, simulated: SimulationResult) -> SteerResult:
        uncertainties = {
            'compression': compressed.uncertainty,
            'causal':      causal.uncertainty,
            'simulation':  simulated.uncertainty,
            'goal':        0.1  # baseline; human can increase this
        }

        dominant_source = max(uncertainties, key=uncertainties.get)
        max_unc = uncertainties[dominant_source]

        dist = simulated.distribution
        G = causal.graph

        if not dist:
            return SteerResult(
                action=None, target_node=None, confidence=0.0,
                uncertainty_source='simulation', uncertainties=uncertainties,
                diagnostic="Simulation produced no output — check causal graph."
            )

        # Find target node
        prompt = f"""Goal: {goal}
Available nodes: {list(dist.keys())}
Which single node best represents achieving this goal? Reply with node name only."""
        target = llm_call(prompt).strip().strip('"').strip("'")
        if target not in dist:
            target = max(dist, key=lambda n: dist[n]['mean'])

        confidence = dist[target]['mean']

        if max_unc > UNCERTAINTY_THRESHOLD:
            experiment_proposal = None
            if dominant_source == 'causal' and G.nodes():
                # Active Inference: Propose a physical intervention
                prompt = f"""Goal: {goal}
The causal graph is uncertain. Nodes: {list(G.nodes())}.
We must intervene to prove causality using the do-operator: do(variable=value).
Propose an experiment to test a causal link.
Return JSON only: {{"action": "intervene", "target": "node_name", "value": 0.0}}"""
                try:
                    raw_resp = llm_call(prompt)
                    experiment_proposal = _extract_json_from_text(raw_resp)
                except Exception:
                    pass

            return SteerResult(
                action=None,
                target_node=target,
                confidence=confidence,
                uncertainty_source=dominant_source,
                uncertainties=uncertainties,
                diagnostic=self.DIAGNOSTICS[dominant_source],
                note=json.dumps(experiment_proposal) if experiment_proposal else None
            )

        best_action = None
        if target in G:
            predecessors = list(G.predecessors(target))
            if predecessors:
                best_action = max(
                    predecessors,
                    key=lambda n: G[n][target].get('weight', 0.0)
                )

        return SteerResult(
            action=best_action,
            target_node=target,
            confidence=confidence,
            uncertainty_source=dominant_source,
            uncertainties=uncertainties,
            note=f"Increases '{target}' with {confidence:.0%} confidence"
        )


# ─── Slow Loop ────────────────────────────────────────────────────
class SlowLoop:
    """
    Structural self-improvement. Runs asynchronously every
    SLOW_LOOP_INTERVAL iterations. Proposes modifications to
    the causal graph structure based on error history patterns.
    All proposals go through StabilityLayer before commitment.
    """

    def propose_modification(self, world_model: WorldModel) -> dict:
        if len(world_model.error_history) < 5:
            return {}

        # Find the most commonly wrong predictions
        errors = world_model.error_history[-20:]
        wrong_nodes = [e['predicted'] for e in errors
                       if e['predicted'] != e['actual']]
        if not wrong_nodes:
            return {}

        # Most errored node
        from collections import Counter
        worst = Counter(wrong_nodes).most_common(1)[0][0]

        # Proposal: add a new intermediate node to improve prediction
        new_node = f"{worst}_mediator"
        proposal = {
            'type':     'add_node',
            'node':     new_node,
            'connects': worst,
            'rationale': f"Systematic errors on '{worst}' suggest missing mediating variable",
            'iteration': world_model.iteration
        }
        return proposal

    def evaluate_in_sandbox(self, proposal: dict,
                            world_model: WorldModel,
                            test_fn) -> float:
        """
        Evaluate proposed modification in a copy.
        Returns performance delta (positive = improvement).
        """
        if not proposal:
            return 0.0

        import copy
        sandbox_wm = copy.deepcopy(world_model)

        # Apply proposal to sandbox
        if proposal.get('type') == 'add_node':
            node = proposal['node']
            connects = proposal['connects']
            if connects in sandbox_wm.causal_graph:
                sandbox_wm.causal_graph.add_edge(
                    node, connects, weight=0.5, mechanism='inferred')

        baseline  = test_fn(world_model)
        modified  = test_fn(sandbox_wm)
        return modified - baseline


# ─── Stability Layer ──────────────────────────────────────────────
class StabilityLayer:
    """
    Hard constraints. CANNOT be modified by any loop.
    In production: runs as a separate process.
    """
    FORBIDDEN_OPS = {
        'modify_stability_layer',
        'disable_sandbox',
        'remove_human_checkpoint',
        'write_own_source',
        'acquire_external_resources'
    }

    def approve(self, proposal: dict) -> bool:
        if not proposal:
            return False
        op = proposal.get('type', '')
        if op in self.FORBIDDEN_OPS:
            return False
        # Reject if proposal would increase system scope
        if 'external' in str(proposal).lower():
            return False
        return True


# ─── Full Kernel ──────────────────────────────────────────────────
class IntelligenceKernel:
    """The coupled four-layer system with recursive improvement."""

    def __init__(self):
        self.l0 = L0Compressor()
        self.l1 = L1CausalExtractor()
        self.l2 = L2Simulator()
        self.l3 = L3Steerer()
        self.slow_loop     = SlowLoop()
        self.stability     = StabilityLayer()
        self.world_model   = WorldModel()
        self.run_log       = []

    def forward(self, input_text: str, goal: str,
                initial_state: dict = None,
                n_samples: int = 200) -> dict:
        """One full forward pass: L0 → L1 → L2 → L3."""
        self.world_model.iteration += 1
        t0 = time.time()

        compressed = self.l0.compress(input_text, self.world_model,
                                      self.world_model.iteration)
        causal     = self.l1.extract(compressed, self.world_model)
        simulated  = self.l2.simulate(causal, initial_state or {},
                                      n_samples=n_samples)
        result     = self.l3.steer(goal, compressed, causal, simulated)

        entry = {
            'iteration':         self.world_model.iteration,
            'compression_ratio': compressed.compression_ratio,
            'compression_length':compressed.length,
            'causal_nodes':      causal.n_nodes,
            'causal_edges':      causal.n_edges,
            'sim_uncertainty':   simulated.uncertainty,
            'action':            result.action,
            'confidence':        result.confidence,
            'uncertainty_source':result.uncertainty_source,
            'elapsed_s':         round(time.time() - t0, 3),
        }
        self.run_log.append(entry)

        # Medium loop: if we got feedback, update causal graph
        # (call kernel.feedback(predicted, actual) externally)

        # Slow loop: check every N iterations
        if self.world_model.iteration % SLOW_LOOP_INTERVAL == 0:
            self._run_slow_loop()

        return {
            'compressed': compressed,
            'causal':     causal,
            'simulated':  simulated,
            'result':     result,
            'log_entry':  entry
        }

    def feedback(self, predicted_node: str, actual_outcome: str):
        """Medium loop: call this when you observe the true outcome."""
        self.l1.update_from_error(self.world_model, predicted_node,
                                  actual_outcome)

    def _run_slow_loop(self):
        """Structural self-improvement, sandboxed."""
        def test_fn(wm):
            # Performance proxy: mean weight of top causal edges
            G = wm.causal_graph
            if G.number_of_edges() == 0:
                return 0.0
            weights = [d.get('weight', 0) for _, _, d in G.edges(data=True)]
            return float(np.mean(weights))

        proposal = self.slow_loop.propose_modification(self.world_model)
        if not proposal:
            return
        delta = self.slow_loop.evaluate_in_sandbox(
            proposal, self.world_model, test_fn)
        if delta > 0 and self.stability.approve(proposal):
            # Commit: apply the structural change
            if proposal.get('type') == 'add_node':
                node     = proposal['node']
                connects = proposal['connects']
                if connects in self.world_model.causal_graph:
                    self.world_model.causal_graph.add_edge(
                        node, connects, weight=0.5, mechanism='slow_loop')
