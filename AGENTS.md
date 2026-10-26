# My Rules

## 通用规则

1. 在需要时,使用skills `miyabi-ml-experiment-platform` , `miyabi-pbs-cmd` , `miyabi-pbs-goal-wakeup` , `miyabi-pbs-shell-template`
2. 优先使用codegraph获取代码库知识.
3. 自行使用uv管理所需依赖, 自行使用git管理代码库状态.
4. 避免在生成代码中使用SHA-256

## 多Agents规则

- 以main agent(当前设定为Claude code的Opus-5.5)为 Orchestrator,对任务进行统筹管理,尽可能考虑任务的并行执行方案; 
- Orchestrator通过`/herdr`在新的tab中开启codex作为worker完成子任务, 选择 GPT-6.1-sol xhigh作为执行模型.
- Agent之间的交流应该以文档为基础,以参考文档+少量描述进行交流; 避免长文本,无留档的交流.
- Orchestrator 和 Worker可以在自己的session中开启子代理(Sonnet-5.5 或 GPT-6.1-sol)
- Codex worker 应该被视为具有固定角色的, 具有长生命周期的subagnet对象. Orchestrator 应该保持并复用codex worker, 在需要时主动清理/重新注入上下文
