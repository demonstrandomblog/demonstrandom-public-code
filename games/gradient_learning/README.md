# Normal-form gradient learning

Companion to [Learning Equilibria by Gradient Descent](https://demonstrandom.com/game_theory/posts/gradient_learning_nash/).

> **AI warning:** This is experimental research code. AI assistance has been
> used in the project and in this cleanup; historical generation details are
> not established for every file. Existing file-level disclosures are retained.
> Passing automated checks does not establish a full human or mathematical
> review. Independently validate the behavior and results you rely on.

## Run

From the repository root:

```sh
python -m pip install -r games/gradient_learning/requirements.txt
python -m games.gradient_learning
python -m pytest -q games/gradient_learning
```

The deterministic example seeds PyTorch with 0, constructs the article's Stag Hunt,
and runs 200 Adam updates. `core` supplies `Game`, `Agent`, `Policy`, `LogitsPolicy`,
and `Arena` for normal-form payoff tensors of shape `(players, actions_1, ...)`.
Create the arena before creating optimizers: it initializes the lazy policy parameters.

```python
import torch
from games.gradient_learning import Agent, Arena, Game, LogitsPolicy

game = Game(torch.tensor([[[8., 0.], [2., 3.]], [[3., 1.], [0., 2.]]]),
            [["Stag", "Hare"], ["Stag", "Hare"]])
arena = Arena(game, [Agent(LogitsPolicy(), "Alice"), Agent(LogitsPolicy(), "Bob")])
print(arena.expected_payoffs())
```

## Article comparison and checks

The recovered training loop computes each player's gradient only with respect to
that player's parameters, with all gradients evaluated before either optimizer
step. This implements the article's simultaneous-update intent; its displayed
loop instead calls `backward()` and steps optimizers in sequence. Random draws
and printed final probabilities differ because the public example fixes a seed.
`Game.clone()` preserves whether payoff parameters require gradients.
The later observation-conditioned policy is outside this article's release.

Four tests check the article's `(4, 2)` payoff example, independent three-player
payoff enumeration, finite-difference payoff gradients, and lazy initialization.

Original code and documentation: [PolyForm Noncommercial 1.0.0](../../LICENSE).
Preserve [NOTICE](../../NOTICE). Citation: [CITATION.cff](../../CITATION.cff).
Dependencies retain their own licenses.
