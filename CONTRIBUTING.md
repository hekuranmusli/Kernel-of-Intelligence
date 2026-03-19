# Contributing to the Intelligence Kernel

First off, thank you for considering contributing to the Intelligence Kernel!

As I mentioned in the README, I am just a curious human experimenting with massive ideas about AGI, causality, and embodiment. This project is a playground, and I would love for you to play in it too.

## How to Contribute

There are no strict, corporate rules here. If you have a wild idea, I want to see it.

Here are some ways you can help:
1. **Robotics & Sensors:** The ultimate goal is to get this running on physical hardware. If you know Arduino, Raspberry Pi, or ROS, and want to hook `embodiment.py` up to real servos and thermal sensors, please submit a Pull Request!
2. **Performance Improvements:** The `L0` and `L1` loops rely heavily on LLM inference. If you know how to distill these loops into smaller, faster models, or how to implement a compiled "System 1", that would be groundbreaking.
3. **New Mystery Boxes:** Can you write a more complex `DigitalEnvironment` in `embodiment.py` to test the Kernel's causal limits?
4. **Bug Fixes:** Found a crash? Open an issue or submit a fix.

## Getting Started

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-wild-idea`)
3. Make your changes
4. Run the experiments (`python experiments.py --fast`) to ensure nothing broke
5. Commit your changes (`git commit -m "Added a thermal sensor intervention"`)
6. Push to the branch (`git push origin feature/your-wild-idea`)
7. Open a Pull Request

## The Golden Rule
Keep the philosophy intact. The system must always prioritize causal inference over correlation, and it must strive to compress its understanding.

Let's build something crazy together!