# Copyright (c) 2024-2026 Kevin T. Procopio and contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI assistance/review status: see AI_NOTICE.md at the repository root.

import torch
from torch import nn, optim
from .core import Agent, Arena, Game, LogitsPolicy

def main():
    torch.manual_seed(0)


    staghunt_actions = [["Stag", "Hare"], ["Stag", "Hare"]]
    p1_payoffs = [
        [8, 0],  # P1 plays Stag vs P2's [Stag, Hare]
        [2, 3]   # P1 plays Hare vs P2's [Stag, Hare]
    ]

    p2_payoffs = [
        [3, 1],  # P1 Stag; columns are P2 Stag/Hare
        [0, 2]   # P1 Hare; columns are P2 Stag/Hare
    ]
    payoffs =  nn.Parameter(torch.tensor([p1_payoffs, p2_payoffs], dtype=torch.float))
    stag_hunt = Game(payoffs, staghunt_actions, "Stag Hunt")
    print(stag_hunt)

    alice = Agent(
        policy=lambda actions: torch.tensor([1.0 if a=="Stag" else 0.0 for a in actions]),
        name="Alice"
    )
    bob = Agent(
        policy=lambda actions: torch.ones(len(actions))/len(actions),
        name="Bob"
    )

    stag_hunt_arena = Arena(stag_hunt, [alice, bob])

    print(stag_hunt_arena.expected_payoffs())
    print(stag_hunt_arena.play())
    print(stag_hunt_arena.play())
    print(stag_hunt_arena.play())


    diff_alice = Agent(
        policy=LogitsPolicy(initialization='random'),
        name="DiffAlice"
    )
    diff_bob = Agent(
        policy=LogitsPolicy(initialization='random'),
        name="DiffBob"
    )
    diff_agents = [diff_alice, diff_bob]
    diff_stag_hunt_arena = Arena(stag_hunt, diff_agents)

    params1 = list(diff_alice.policy.parameters())
    params2 = list(diff_bob.policy.parameters())

    opt1 = optim.Adam(params1, lr=0.1)
    opt2 = optim.Adam(params2, lr=0.1)

    for step in range(200):
        exp = diff_stag_hunt_arena.expected_payoffs()
        loss1 = -exp[0]
        loss2 = -exp[1]

        opt1.zero_grad(set_to_none=True)
        opt2.zero_grad(set_to_none=True)

        g1 = torch.autograd.grad(loss1, params1, retain_graph=True)
        g2 = torch.autograd.grad(loss2, params2)

        for p, g in zip(params1, g1): p.grad = g
        for p, g in zip(params2, g2): p.grad = g

        opt1.step()
        opt2.step()

        if step % 20 == 0:
            print(f"Step {step}, Expected Payoffs: {exp.detach().numpy()}")

    # Final Policies
    for i, agent in enumerate(diff_agents):
        logits = agent.policy.logits.detach().numpy()
        probs = agent.policy(stag_hunt.actions[i]).detach().numpy()
        print(f"Agent {i+1} final logits: {logits}")
        print(f"Agent {i+1} final probabilities: {probs}")
        print(f"Agent {i+1} prefers: {'Stag' if probs[0] > probs[1] else 'Hare'}")
        print()



if __name__ == "__main__":
    main()
