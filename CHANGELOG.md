# Changelog

## v0.1.9

### Added

- **Interrupt mode (human-in-the-loop):** Press Ctrl+C during an agent run to pause after the current operation completes, type a message to inject into the conversation, and resume. Double Ctrl+C for hard abort. Enabled by default for CLI runs; disable with `--disable-interrupt`.
- `--disable-interrupt` CLI flag for `prophet run`.
- `enable_interrupt` config field in `CliAgentConfig`.

### Changed

- Extracted `_prepare_messages_for_step()` from `DefaultForecastAgent.step()` for cleaner subclass overrides.

## v0.1.8

- Add token usage tracking and context window monitoring.
- Add grace period feature to `DefaultForecastAgent`.
- Add Tavily search backend integration.
- Fix sliding window context manager discarding assistant message with multiple tool calls.
- Augment batch process to accept custom model.
- Update docs to match current code.
