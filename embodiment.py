import json

class DigitalEnvironment:
    """Base class for an embodied interventional environment."""
    def __init__(self):
        self.state = {}
        self._interventions = {}

    def observe(self):
        """Returns the current state of the universe."""
        return self.state.copy()

    def intervene(self, variable, value):
        """
        Judea Pearl's do-operator: do(X=x)
        Forces a variable to a specific value, severing its incoming causal links.
        """
        if variable in self.state:
            self._interventions[variable] = float(value)
            self.state[variable] = float(value)
            return f"SUCCESS: do({variable}={value})"
        return f"ERROR: Unknown variable {variable}"

    def release(self, variable):
        """Stops intervening on a variable, letting it return to natural causal flow."""
        if variable in self._interventions:
            del self._interventions[variable]
            return f"RELEASED: {variable} is now under natural causal control."
        return f"ERROR: Not intervening on {variable}"

    def step(self):
        """Advances time by 1 tick, applying hidden causal rules. Must be overridden."""
        raise NotImplementedError


class MysteryBox(DigitalEnvironment):
    """
    A black-box environment with hidden causal rules.
    The Kernel must discover these rules through intervention, not reading text.

    Hidden Causal Graph (The Ground Truth the Kernel must discover):
    [Switch] -> [Heater] -> [Temperature] -> [Pressure]
    """
    def __init__(self):
        super().__init__()
        # Initial State
        self.state = {
            "switch": 0.0,       # 0 (off) or 1 (on)
            "heater": 0.0,       # 0 (off) or 1 (on)
            "temperature": 20.0, # baseline 20C
            "pressure": 30.0     # baseline 30 psi
        }

    def step(self):
        """The Physics Engine of the Mystery Box."""

        # Rule 1: Switch activates Heater (unless Heater is being physically overridden)
        if "heater" not in self._interventions:
            self.state["heater"] = self.state["switch"]

        # Rule 2: Heater changes Temperature (unless Temperature is physically overridden)
        if "temperature" not in self._interventions:
            if self.state["heater"] > 0.5:
                self.state["temperature"] += 5.0  # Heats up by 5 degrees per tick
            else:
                self.state["temperature"] = max(20.0, self.state["temperature"] - 2.0) # Cools down to room temp

        # Rule 3: Temperature strictly causes Pressure (unless Pressure is overridden)
        # Gay-Lussac's Law approximation
        if "pressure" not in self._interventions:
            self.state["pressure"] = 30.0 + (self.state["temperature"] - 20.0) * 1.5

        return self.observe()


if __name__ == "__main__":
    print("--- Booting Digital Embodiment Environment: The Mystery Box ---\n")
    env = MysteryBox()

    print("--- 1. OBSERVATIONAL PHASE (Just watching) ---")
    print(f"Initial State: {json.dumps(env.observe(), indent=2)}")

    print("\nAction: Human turns on the 'switch'.")
    env.intervene("switch", 1.0)

    for i in range(3):
        print(f"Tick {i+1}: {env.step()}")

    print("\n[Kernel's Hypothesis]: Switch, Heater, Temperature, and Pressure all go up together.")
    print("Is it Temperature that causes Pressure? Or does Pressure cause Temperature? Correlation cannot tell us.\n")

    print("--- 2. INTERVENTIONAL PHASE (The 'do' operator) ---")
    print("Action: Kernel autonomously executes do(temperature = 0.0).")
    print("(Imagine a robot opening a window to freeze the room, despite the heater being on)")

    env.intervene("temperature", 0.0)

    for i in range(3):
        print(f"Tick {i+1}: {env.step()}")

    print("\n[Kernel's Epiphany]:")
    print("Even though the Heater is ON (1.0), the Pressure CRASHED to 0.0 because I forced Temperature to 0.0.")
    print("CONCLUSION: Heater -> Temperature -> Pressure. Pressure DOES NOT cause Temperature.")
    print("The Kernel has just learned true physics by intervening in the world!")
