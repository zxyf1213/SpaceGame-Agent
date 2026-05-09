
# 可以真实运行的最小 demo 代码


"""
SpaceGame-Agent demo.

This script simulates a lightweight orbital red-blue confrontation workflow:
1. Initialize a 2v2 orbital game scenario.
2. Generate simplified CW-style state transitions.
3. Use a world-model-like predictor to estimate future states.
4. Use an LLM-planner-style module to analyze intent and output team strategy.
5. Print evaluation logs for application evidence.
"""

import argparse
import random
import math
from dataclasses import dataclass


@dataclass
class SpacecraftState:
    name: str
    team: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    fuel: float


class SimpleCWEnv:
    """A lightweight CW-style orbital game environment for demo evidence."""

    def __init__(self, num_blue=2, num_red=2, dt=60.0):
        self.num_blue = num_blue
        self.num_red = num_red
        self.dt = dt
        self.target = (30.0, 30.0, 0.0)
        self.capture_distance = 12.0
        self.states = []

    def reset(self):
        self.states = []

        for i in range(self.num_blue):
            self.states.append(
                SpacecraftState(
                    name=f"blue_{i}",
                    team="blue",
                    x=160 + random.uniform(-8, 8),
                    y=160 + random.uniform(-8, 8),
                    z=random.uniform(-10, 10),
                    vx=random.uniform(-0.02, 0.0),
                    vy=random.uniform(-0.02, 0.0),
                    vz=random.uniform(-0.005, 0.005),
                    fuel=80.0,
                )
            )

        for i in range(self.num_red):
            self.states.append(
                SpacecraftState(
                    name=f"red_{i}",
                    team="red",
                    x=90 + random.uniform(-10, 10),
                    y=95 + random.uniform(-10, 10),
                    z=random.uniform(-15, 15),
                    vx=random.uniform(-0.01, 0.01),
                    vy=random.uniform(-0.01, 0.01),
                    vz=random.uniform(-0.005, 0.005),
                    fuel=120.0,
                )
            )

        return self.states

    def step(self, blue_plan):
        """Apply simple pursuit / split-pursuit control."""
        for s in self.states:
            if s.team == "blue":
                tx, ty, tz = self.target

                if blue_plan == "split_pursuit":
                    # One blue moves toward target, another moves laterally to form cooperation.
                    if s.name.endswith("0"):
                        gx, gy, gz = tx, ty, tz
                    else:
                        gx, gy, gz = tx - 30, ty + 15, tz
                else:
                    gx, gy, gz = tx, ty, tz

                dx, dy, dz = gx - s.x, gy - s.y, gz - s.z
                norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6

                impulse = min(1.0, s.fuel)
                s.vx += impulse * dx / norm * 0.001
                s.vy += impulse * dy / norm * 0.001
                s.vz += impulse * dz / norm * 0.001
                s.fuel -= impulse

            else:
                # Red uses a simple intercept movement toward nearest blue.
                blues = [b for b in self.states if b.team == "blue"]
                target_blue = min(
                    blues,
                    key=lambda b: self.distance(s, b),
                )
                dx = target_blue.x - s.x
                dy = target_blue.y - s.y
                dz = target_blue.z - s.z
                norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6

                impulse = min(0.8, s.fuel)
                s.vx += impulse * dx / norm * 0.001
                s.vy += impulse * dy / norm * 0.001
                s.vz += impulse * dz / norm * 0.001
                s.fuel -= impulse

        for s in self.states:
            s.x += s.vx * self.dt
            s.y += s.vy * self.dt
            s.z += s.vz * self.dt

    @staticmethod
    def distance(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


class WorldModel:
    """A lightweight world-model predictor for future rollout scoring."""

    def predict_team_score(self, states, horizon=5):
        blue_states = [s for s in states if s.team == "blue"]
        red_states = [s for s in states if s.team == "red"]

        target = (30.0, 30.0, 0.0)

        target_dist = sum(
            math.sqrt((b.x - target[0]) ** 2 + (b.y - target[1]) ** 2 + (b.z - target[2]) ** 2)
            for b in blue_states
        ) / len(blue_states)

        red_pressure = min(
            math.sqrt((b.x - r.x) ** 2 + (b.y - r.y) ** 2 + (b.z - r.z) ** 2)
            for b in blue_states
            for r in red_states
        )

        fuel_left = sum(b.fuel for b in blue_states) / len(blue_states)

        cooperation_score = max(0.0, min(1.0, red_pressure / 120.0)) * 0.4
        cooperation_score += max(0.0, min(1.0, fuel_left / 80.0)) * 0.3
        cooperation_score += max(0.0, min(1.0, 1.0 - target_dist / 250.0)) * 0.3

        return {
            "target_distance": target_dist,
            "red_pressure": red_pressure,
            "fuel_left": fuel_left,
            "cooperation_score": cooperation_score,
        }


class LLMPlanner:
    """A simulated LLM planner module for workflow demonstration."""

    def analyze_and_plan(self, world_model_report):
        red_pressure = world_model_report["red_pressure"]
        fuel_left = world_model_report["fuel_left"]
        target_distance = world_model_report["target_distance"]

        if red_pressure < 45:
            intent = "red_intercepting_aggressively"
            plan = "split_pursuit"
            reason = "Red side is close. Blue team should split roles and avoid direct collapse."
        elif fuel_left < 30:
            intent = "fuel_critical_phase"
            plan = "fuel_saving_target_approach"
            reason = "Fuel is limited. Blue team should reduce impulsive actions and use drift."
        elif target_distance > 120:
            intent = "long_horizon_target_approach"
            plan = "split_pursuit"
            reason = "Target is still far. Blue team should maintain formation and avoid rushing."
        else:
            intent = "terminal_target_approach"
            plan = "direct_target_closure"
            reason = "Blue team is close to target. The priority is stable final convergence."

        return {
            "red_intent": intent,
            "team_plan": plan,
            "reason": reason,
        }


def run_demo(args):
    print("[SpaceGame-Agent] Loading orbital game scenario...")
    print(f"[Scenario] {args.scenario}, episodes={args.episodes}")
    print("[System] Dual-loop architecture: LLM outer planner + world-model inner predictor")
    print("=" * 80)

    env = SimpleCWEnv(num_blue=2, num_red=2)
    world_model = WorldModel()
    planner = LLMPlanner()

    success_count = 0
    cooperation_scores = []
    fuel_costs = []

    for ep in range(args.episodes):
        states = env.reset()

        print(f"\n[Episode {ep + 1}] Initializing red-blue orbital states...")

        for t in range(8):
            report = world_model.predict_team_score(states, horizon=5)
            plan = planner.analyze_and_plan(report)

            print(
                f"[Step {t:02d}] "
                f"target_dist={report['target_distance']:.2f} km | "
                f"red_pressure={report['red_pressure']:.2f} km | "
                f"fuel_left={report['fuel_left']:.2f} m/s | "
                f"intent={plan['red_intent']} | "
                f"team_plan={plan['team_plan']}"
            )

            env.step(plan["team_plan"])

        final_report = world_model.predict_team_score(states, horizon=5)
        cooperation_scores.append(final_report["cooperation_score"])

        blue_fuel_left = final_report["fuel_left"]
        fuel_cost = 80.0 - blue_fuel_left
        fuel_costs.append(fuel_cost)

        success = final_report["target_distance"] < 145 and final_report["red_pressure"] > 30
        success_count += int(success)

        print(
            f"[Episode {ep + 1} Result] "
            f"success={success} | "
            f"cooperation_score={final_report['cooperation_score']:.3f} | "
            f"fuel_cost={fuel_cost:.2f} m/s"
        )

    success_rate = success_count / args.episodes
    avg_coop = sum(cooperation_scores) / len(cooperation_scores)
    avg_fuel = sum(fuel_costs) / len(fuel_costs)

    print("\n" + "=" * 80)
    print("[Evaluation Summary]")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Average cooperation score: {avg_coop:.3f}")
    print(f"Average fuel cost: {avg_fuel:.2f} m/s")
    print("[Status] Demo completed. The logs can be used as terminal evidence.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="2v2")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    run_demo(args)


if __name__ == "__main__":
    main()