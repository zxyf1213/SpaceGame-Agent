# SpaceGame-Agent

SpaceGame-Agent is a research prototype for LLM-assisted multi-agent orbital game decision-making.

The project focuses on red-blue orbital confrontation under CW relative dynamics. It combines:

- Multi-agent orbital simulation
- Pulse-based spacecraft control
- Transformer-style world model planning
- LLM-based intent recognition
- Automatic trajectory failure analysis
- Agent-driven reward and policy debugging

## Core Problem

Traditional rule-based policies and single-stage RL policies struggle in orbital confrontation scenarios because they need to handle long-horizon planning, fuel constraints, multi-agent cooperation, partial observability, and adversarial maneuvers simultaneously.

## System Architecture

The system uses a dual-loop design:

1. **Outer LLM Agent**
   - Reads trajectory logs
   - Identifies red-side intent
   - Diagnoses failure causes
   - Generates high-level team plans
   - Suggests reward and policy modifications

2. **Inner RL / World Model Agent**
   - Simulates CW orbital dynamics
   - Predicts future trajectories
   - Evaluates candidate joint actions
   - Produces executable control decisions

## Demo

```bash
python run_demo.py --scenario 2v2 --episodes 3
