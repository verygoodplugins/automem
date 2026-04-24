# Benchmark Judge Policy

AutoMem internal benchmark runs use a pinned OpenAI judge snapshot:

- Model: `gpt-5.4-mini-2026-03-17`
- Profile: `openai-gpt-5.4-mini-2026-03-17`
- Provider: `openai`

Use this judge for routine LoCoMo, LongMemEval, and regression tracking so
scores remain comparable over time. Result JSON should record `judge_model`,
`judge_profile`, `judge_provider`, and `judge_snapshot_pinned`.

External published comparisons are separate runs. Match the published judge
configuration exactly and label the run with an explicit non-canonical profile
such as `published-mem0-gpt-5`. Do not compare scores unless judge metadata
matches or the difference is called out directly.
