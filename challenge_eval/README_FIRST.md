# Start here

1. Extract this ZIP in your Listing Lens project root. It creates `challenge_eval/`.
2. Collect a fresh, independent challenge set. Suggested target: GPT 50, Gemini 50, Meta AI 50 and 100 verified genuine photographs.
3. Place files under `challenge_eval/data/real/` or `challenge_eval/data/ai_generated/<generator>/`.
4. Record provenance in `challenge_eval/data/manifest.csv` and prompts in `challenge_eval/data/prompt_log.csv` (copy the provided templates).
5. Do not use these images in training or validation.
6. Run the command in `challenge_eval/README.md`.

No new training occurs in this stage. Both V3 and V4 weights remain unchanged.
