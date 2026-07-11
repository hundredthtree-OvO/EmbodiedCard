---
name: build-embodied-learning-cards
description: Generate, deepen, compare, validate, and organize progressive Markdown learning cards for embodied intelligence concepts, especially VLA, WAM/world models, robot learning, action generation, representation learning, reinforcement learning, planning, control, and their mathematical or deep-learning prerequisites. Use when the user wants to learn a module such as VAE, ELBO, flow matching, diffusion, action chunking, RSSM, attention, value functions, SE(3), or related technology through structured L0–L3 cards with math, diagrams, PyTorch, embodied-system context, self-tests, sources, an index, or a knowledge dependency graph.
---

# Build Embodied Learning Cards

Create durable, technically rigorous learning cards that explain not only what a concept is, but why it exists, how it works, and where it belongs in a VLA/WAM or robot-learning system.

## Route the request

- **Create**: Generate one concept card at L0–L2 by default.
- **Deepen**: Read the existing card and add only the requested L3 material. Preserve user edits.
- **Create prerequisites**: Split a derivation or dependency into atomic child cards and link them from the main card.
- **Compare**: Create a comparison card only when the comparison itself is the learning objective; otherwise keep the comparison table in the main concept card.
- **Quiz**: Generate questions from existing cards without inventing facts outside them unless the user requests research.
- **Organize**: Update `INDEX.md` and `KNOWLEDGE_GRAPH.md` when the user is maintaining a card collection.

## Build one card

1. Define one core concept per `.md` file. Use a main card plus atomic child cards for substantial derivations, failure modes, or prerequisites.
2. Determine the user's likely knowledge boundary from the request and existing cards. Explain prerequisites briefly; link to a child card when the explanation would interrupt the main line.
3. Research before writing when claims depend on a paper, implementation, current method, or exact architecture. Follow `references/source-policy.md`.
4. Start from `assets/card-template.md`. Include L0–L2 unless the user requests another depth. Leave a concise L3 backlog rather than expanding it automatically.
5. Connect the concept to concrete embodied uses. Distinguish a classical module from descendants that merely share its principles; for example, do not call RSSM a plain VAE.
6. Include minimal PyTorch when implementation helps understanding. State the likelihood or modeling assumptions behind a loss rather than presenting MSE/BCE as arbitrary choices.
7. Add compact Mermaid only when structure, sequence, or dependency is easier to understand visually. Always follow a diagram with a one-sentence text equivalent.
8. Add self-tests at memory, understanding, and transfer levels. Put answers in standard HTML `<details>` blocks compatible with GitHub.
9. Add primary references and label inferences as inferences.
10. Run `python3 scripts/validate_card.py <card-or-directory>` and fix every error before delivery.

## Enforce Markdown and math portability

Target standard VS Code Markdown Preview and GitHub Markdown, not Obsidian syntax.

- Use `[label](relative/path.md)`, never `[[Wiki Links]]`.
- Use `$...$` for inline math.
- Use a standalone `$$` line before and after display math.
- Never use `\(...\)` or `\[...\]`; these are inconsistently handled and may lose backslashes in file-transfer paths.
- Keep a blank line before and after every display-math block.
- Do not place display math inside tables. Use short inline math in cells and move long formulas below the table.
- Avoid tabs and unescaped control characters in formulas.
- Use `\begin{aligned}...\end{aligned}` only inside `$$` delimiters.
- Prefer LaTeX commands such as `\mu`, `\theta`, and `\mathcal{L}` inside math delimiters; Unicode symbols are acceptable in diagrams and prose.
- Mention that VS Code's built-in preview must have `markdown.math.enabled` enabled if valid `$`/`$$` formulas still do not render.

## Preserve pedagogical order

Use this sequence:

1. **L0 — orientation**: one-sentence definition, problem solved, embodied role, three takeaways.
2. **L1 — mental model**: predecessor's limitation, intuition, architecture/data flow, inputs and outputs, VLA/WAM placement, nearby-method comparison.
3. **L2 — mechanism**: notation, core equations, derivation at the appropriate depth, numeric example, tensor shapes, training versus inference, pseudocode, PyTorch, formula-to-code mapping, hyperparameters, failure modes, misconceptions.
4. **Navigation**: prerequisites, atomic child cards, comparisons, and next card.
5. **L3 backlog**: paper-level derivations, source mapping, variants, and open research questions to add later.

Do not repeat the same explanation at every level. Each level must add a new kind of understanding.

## Manage the card collection

Use the taxonomy in `references/card-taxonomy.md` as a starting point, adapting it to the user's existing repository.

- Keep filenames stable and readable, such as `VAE.md`, `Flow-Matching.md`, or `SE3.md`.
- Store atomic cards near their main card when this keeps relative links simple.
- Record prerequisites and related concepts in YAML frontmatter.
- Update `INDEX.md` with the card title, path, level, status, and one-line purpose.
- Update `KNOWLEDGE_GRAPH.md` with small topic-specific Mermaid graphs. Do not create one unreadable global graph.
- When updating an existing card, patch only the affected sections and update the `updated` date.

## Quality gate

Read `references/quality-checklist.md` before finalizing a new template style or a high-stakes mathematical card. At minimum verify:

- symbols are defined before use;
- posterior, prior, likelihood, loss, and inference-time behavior are not conflated;
- equations and code use the same reduction and sign conventions;
- numeric examples reproduce the stated result;
- local links resolve;
- external claims have authoritative sources;
- Markdown contains no Obsidian-only constructs;
- validation passes.

## Resources

- `assets/card-template.md`: GitHub/VS Code-compatible L0–L3 card skeleton.
- `references/source-policy.md`: research and citation priority.
- `references/card-taxonomy.md`: default VLA/WAM learning-card organization.
- `references/quality-checklist.md`: pedagogical and technical review checklist.
- `scripts/validate_card.py`: deterministic structural, math-delimiter, link, and Python-block validation.
