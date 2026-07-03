# RAG Hardening Update

Date: `2026-06-25`

This update tightens three behavior gaps in the recipe RAG system.

## What Changed

### 1. Topic switches now perform a real session reset

- When a user says phrases like `换个话题`, the active session no longer keeps old conversation history in prompt context.
- The current dish entity and intent are cleared together with prior turn history for that session state.

### 2. Streaming answers now use the same conversation-aware path as non-streaming answers

- Streaming detail/general responses now go through the conversation-aware generation flow.
- Follow-up questions keep prior context consistently in both CLI streaming mode and the Flask SSE endpoint.
- This removes the previous mismatch where non-streaming had multi-turn context but streaming often behaved like a stateless request.

### 3. Shared in-memory session state is now hardened for concurrent access

- Conversation state now uses thread-safe locking around session creation, reset, reads, writes, and expiry cleanup.
- Recommendation cache access in generation state is also guarded so concurrent web requests do not race on shared dictionaries.

## Tests Added

- Session topic-switch regression test
- Streaming conversation-context regression test
- Parallel access safety test for the conversation manager

## Impact

- Better isolation between unrelated topics
- More predictable follow-up answers in SSE/streaming mode
- Lower risk of session cross-talk under threaded Flask traffic
