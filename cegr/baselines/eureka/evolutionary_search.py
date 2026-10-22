from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    code: str
    generation: int
    fitness: float = 0.0  # task success rate from RL probe
    reflection: str = ""
    rl_episodes: int = 0
    info: Dict[str, Any] = field(default_factory=dict)


class EvolutionarySearch:
    """
    Manages Eureka's evolutionary population:
      - Population size K = 2 candidates per generation
      - Generations G = 2 (Total LLM calls = K * G = 4)
      - Elitism selection: picks best candidate from Gen 1 as parent for Gen 2
    """

    def __init__(self, pop_size: int = 2, max_generations: int = 2):
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.generations: Dict[int, List[Candidate]] = {}

    def add_candidate(self, gen: int, cand: Candidate):
        if gen not in self.generations:
            self.generations[gen] = []
        self.generations[gen].append(cand)

    def select_elite(self, gen: int) -> Optional[Candidate]:
        cands = self.generations.get(gen, [])
        if not cands:
            return None
        return max(cands, key=lambda c: c.fitness)

    def get_best_overall(self) -> Optional[Candidate]:
        all_cands = [c for gen_list in self.generations.values() for c in gen_list]
        if not all_cands:
            return None
        return max(all_cands, key=lambda c: c.fitness)
