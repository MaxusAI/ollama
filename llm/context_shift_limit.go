package llm

// ContextShiftPromptLimit is the length llama-server's path truncates an
// over-long prompt to (completionPromptForRequest). The MLX runner cuts to the
// same length under an explicit num_ctx, so the two runners agree on how much
// of a prompt survives (ADR 0050).
func ContextShiftPromptLimit(numCtx, numKeep int) int {
	return contextShiftPromptLimit(numCtx, numKeep)
}
