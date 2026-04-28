# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Personal study notes for a course on Explainable Machine Learning (XAI). The repo contains **only Markdown** — no source code, build system, tests, or dependencies. Tasks here are about reading, organizing, summarizing, or editing notes, not running software.

## Structure

Three top-level modules, each split into numbered sections; each section is a folder of numbered `.md` files that are read in order. The numeric prefixes (`1.`, `2.`, …) define the intended reading order — preserve them when adding or renaming files.

- `Module 01: Model Agnostic Explainability/` — XAI fundamentals, then local methods (LIME, Anchors, Shapley/SHAP, ICE), global methods (functional decomposition, PFI, PDP, ALE), and example-based methods (prototypes, counterfactuals, influential instances).
- `Module 02: Explainable Deep Learning/` — feature visualization & attribution, network dissection, concept activation vectors, attention-based explanations vs. saliency.
- `Module 03: Explainable Generative AI/` — XAI for LLMs (fine-tuning, prompting, RAG) and for generative CV (GANs, diffusion).

Folder and file names contain spaces and colons — always quote paths in shell commands.

## Editing conventions

- The notes are **transcribed from lecture audio** and contain artifacts: zero-width spaces (`U+200B`) at the start of many lines and `[MUSIC]` markers. Don't strip these wholesale unless the user asks — they're part of the existing files and an unrelated edit shouldn't reformat them.
- When matching text for `Edit`, the leading invisible character matters; copy the exact bytes from `Read` output rather than retyping the line.
- Each note file starts with a single `# Title` H1 matching the filename (without the numeric prefix).

## Git

Commits so far follow the form `Added all module N notes` / `Initial Commit`. Match that style for bulk additions; use a descriptive message for targeted edits.
