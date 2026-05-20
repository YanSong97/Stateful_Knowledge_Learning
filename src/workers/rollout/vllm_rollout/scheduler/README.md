# Scheduler

The design of `scheduler` is to decouple the agent workflow from the agent-LLM completion. The CompletionCallback is the definition of agent workflow where it has access to scheduler completion function that returns LLM's response during the process.

For single Tool-augmented reasoning, the workflow is easily desinged as 

