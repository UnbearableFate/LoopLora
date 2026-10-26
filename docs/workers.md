# Worker sessions (for resume: `codex resume <session-id>` in the listed worktree)

Model: workers started before 2026-10-26 20:00 JST run GPT-6-Astra xhigh; new sessions use GPT-6.1-sol xhigh
(`herdr agent start <name> --kind codex --pane <pane> -- -m gpt-6.1-sol -c model_reasoning_effort=xhigh`).

| worker | WP | worktree / branch | codex session | status |
| --- | --- | --- | --- | --- |
| wk-core | WP-A core | LoopLora-wt/core, wp/core | 01a1779d-747b-7df1-8492-b517a9be8c6a | done, merged b367d3a, closed |
| wk-data | WP-B data | LoopLora-wt/data, wp/data | 01a1779d-96b3-7431-8fb7-0458e768764e | done, merged 678ab7c, closed |
| wk-llm | WP-E llm → WP-G | LoopLora-wt/llm, wp/llm | 01a1779d-c123-75a1-9458-ae849ee0fcde | WP-E merged c88485f; now WP-G |
| wk-sym | WP-D sym → WP-D2 | LoopLora-wt/sym, wp/sym | 01a177a0-f742-7c63-8800-0983d801fc19 | WP-D merged c86d72b; now WP-D2 pilots |
| wk-theory | WP-T theory | LoopLora-wt/theory, wp/theory | 01a177c4-57fb-7f42-a920-533e74680115 | done, merged d897f02, closed |
