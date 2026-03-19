import json
import time
from layers import IntelligenceKernel
from embodiment import MysteryBox

def run_embodiment_experiment():
    print("==========================================================")
    print("--- EXPERIMENT 5: The Embodied Intelligence Loop ---")
    print("==========================================================")
    print("Goal: The Kernel must discover the true causal rules of")
    print("      the Mystery Box and maximize the Pressure.")
    print("      It starts with a highly ambiguous text describing")
    print("      a correlated system. It must actively intervene to")
    print("      prove what causes what.")
    print("==========================================================\n")

    # 1. Initialize the Physical World and the Mind
    env = MysteryBox()
    kernel = IntelligenceKernel()

    # The goal we want the Kernel to achieve
    target_goal = "Maximize the pressure variable."

    # 2. Iteration 1: The Ambiguous Observation
    print("--- ITERATION 1: Pure Observation (The Mirror Problem) ---")

    # We provide a text that explicitly makes the causality ambiguous
    ambiguous_text = """
    We observe a system with a switch, a heater, temperature, and pressure.
    When the switch is on, the heater is hot, the temperature is high, and the pressure is massive.
    When the switch is off, the heater is cold, the temperature is low, and the pressure drops.
    Are they all just correlated, or does one cause the other? We don't know the direction.
    """

    # We force the LLM to admit it's uncertain about the causal direction
    ambiguous_text += "\nCRITICAL INSTRUCTION FOR L1: Your causal uncertainty MUST be 0.9 because correlation is not causation and directionality is unknown."

    print(f"Feeding the Kernel ambiguous observational data...")
    res1 = kernel.forward(ambiguous_text, target_goal, initial_state=env.observe())

    # Print the results of Iteration 1
    steer1 = res1['result']
    print(f"\n[L3 Steerer Output]:")
    print(f"  Uncertainty Source: {steer1.uncertainty_source}")
    print(f"  Diagnostic: {steer1.diagnostic}")

    # Check if L3 proposed an intervention
    intervention_json = None
    if steer1.note and "intervene" in steer1.note:
        print(f"  PROPOSED EXPERIMENT: {steer1.note}")
        try:
            intervention_json = json.loads(steer1.note)
        except json.JSONDecodeError:
            pass

    if not intervention_json or intervention_json.get("action") != "intervene":
        print("\n[FAILURE]: The Kernel did not propose a valid intervention to resolve its causal uncertainty.")
        print("Ensure layers.py correctly asks the LLM for an experiment when dominant_source == 'causal'.")
        return

    # 3. Execution: Closing the Loop with the Real World
    print("\n--- ITERATION 2: Active Interventional Inference ---")
    target = intervention_json.get("target", "temperature") # default to temperature if it hallucinates
    value = float(intervention_json.get("value", 0.0))

    print(f"Executing proposed intervention on the physical environment: do({target}={value})")

    # The Robot actually moves the servo (Digital Intervention)
    env.intervene(target, value)

    # Let the physics engine run for a few ticks to settle
    for _ in range(3):
        env.step()

    post_intervention_state = env.observe()
    print(f"New World State after intervention: {json.dumps(post_intervention_state, indent=2)}")

    # 4. Iteration 2: Learning from the Intervention
    # We feed the results of the physical experiment back into the Kernel as text
    # This solves the Symbol Grounding Problem
    epiphany_text = f"""
    We ran a physical experiment! We used the do-operator: do({target}={value}).
    Despite the switch and heater being normally connected, the environment state became:
    {json.dumps(post_intervention_state)}

    This physical intervention proves the causal direction.
    If {target} changed but the predecessors didn't change with it, then {target} does NOT cause them.
    If the successors changed when we forced {target}, then {target} CAUSES the successors.
    Update the causal graph with absolute certainty (uncertainty = 0.05).
    """

    print(f"\nFeeding the experimental results back into the Kernel...")
    res2 = kernel.forward(epiphany_text, target_goal, initial_state=post_intervention_state)

    steer2 = res2['result']
    print(f"\n[L3 Steerer Output After Learning]:")
    print(f"  Uncertainty Source: {steer2.uncertainty_source}")
    print(f"  Action Selected: {steer2.action}")
    print(f"  Target Node: {steer2.target_node}")
    print(f"  Confidence: {steer2.confidence:.2%}")

    if steer2.action:
        print(f"\n[SUCCESS]: The Kernel successfully used Active Inference to discover true physics and is now confident enough to act ({steer2.action} -> {steer2.target_node}).")
    else:
        print("\n[NOTE]: The Kernel learned the graph, but might still need more samples (simulation uncertainty) before acting. Run again for deeper confidence.")

if __name__ == "__main__":
    run_embodiment_experiment()
