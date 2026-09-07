# Opinion Model

## Opinion-Leader Branch

`opleader` develops a two-step-flow-inspired agent-based model of interpersonal message diffusion after information about a focal issue has already entered the social system.

The current model contains opinion-leader and ordinary agents in a fixed social network. Every agent may originate a message, with leaders receiving an origination advantage. Originated messages are delivered deterministically through regular social ties, retain their source identity, and affect recipients only in the next synchronous state update.

Current design documents:

- [Mass-Communication Opinion-Leader Model: Rough Design](docs/mass-communication-opinion-leader-rough-design.md)
- [Posting Origination](docs/posting-origination.md)
- [Message Selection](docs/message-selection.md)
- [Message Aggregation](docs/message-aggregation.md)

## Quick Check

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```
