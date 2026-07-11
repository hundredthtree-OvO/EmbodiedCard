# Quality checklist

## Teaching

- Does L0 answer what, why, and where without unexplained equations?
- Does L1 state what the predecessor already solves before exposing what remains unsolved?
- Does the card follow `background → limitation → design goal → mechanism → new cost` instead of listing mechanisms in parallel?
- Is every major equation introduced by the concrete question or requirement it answers?
- For multi-stage systems, are training stages and deployment/generation flows separated before local details?
- Does L2 connect notation, equations, shapes, training, inference, and code?
- Is a substantial derivation split into an atomic child card?
- Do transfer questions require applying the idea to VLA/WAM rather than repeating definitions?

## Mathematics

- Are all symbols defined before use?
- Are expectations annotated with the distribution being sampled?
- Are maximization objectives and minimized losses distinguished by sign?
- Are prior, approximate posterior, and true posterior distinguished?
- Are per-sample, per-dimension, sum, and mean reductions explicit?
- Is every key objective connected through assumption, simplification or estimator, tensor operation, and reduction?
- Does the prose distinguish exact equality, equality up to constants or scale, Monte Carlo estimates, and engineering surrogates?
- Was every numeric example independently recalculated?
- Are all inline formulas delimited by `$` and display formulas enclosed by fenced `math` blocks?

## Code

- Does the PyTorch code implement the exact convention used by the equations?
- Is every MSE, BCE, cross-entropy, straight-through operation, and sampled expectation mathematically motivated near the code?
- Does the formula-to-code mapping state the modeling assumption and output shape/reduction rather than only pair names?
- Are tensor shapes stated?
- Is the code minimal enough to expose the mechanism?
- Was each Python block syntax-checked and executed when dependencies are available?
- If execution was impossible, is that limitation disclosed?

## Embodied relevance

- Is the module placed in a concrete observation-to-action or world-model pipeline?
- Is training-time information distinguished from deployment-time information?
- Are open-loop/closed-loop and single-action/action-chunk differences stated when relevant?
- Are classical concepts distinguished from VLA/WAM-specific descendants?

## Files

- Do relative links resolve?
- Does the card avoid Obsidian-only syntax?
- Do Mermaid blocks remain compact and have text equivalents?
- Are sources primary and direct?
- Did `scripts/validate_card.py` pass?
