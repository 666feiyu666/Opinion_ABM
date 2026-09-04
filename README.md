# Opinion Model

## Opinion-Leader Branch

`opleader` extends the shared contracts and null baseline with a mass-communication opinion-leader case inspired by Katz's two-step-flow research.

The branch represents communication among the press, opinion leaders, and ordinary agents before the introduction of an algorithmic online platform.

## Goal of the Opinion-Leader Branch

Establish a minimal model in which opinion leaders mediate influence between mass media and a less-active audience.

The model should contain:

- a non-adaptive press that produces externally specified messages;
- adaptive opinion leaders that receive, update, and produce topic messages;
- adaptive ordinary agents that receive and update but do not produce topic messages;
- a stable directed communication network with no tie formation or removal;
- probabilistic message delivery through the fixed communication structure;
- wider interpersonal reach for opinion leaders;
- source identities retained by messages;
- source-dependent evidence weights;
- an explicit rule for aggregating multiple messages received in one round;
- a synchronous update schedule in which messages affect an agent's next state;
- no same-round effect of newly received information on leader production;
- no algorithmic platform preference or engagement feedback.

The model should keep leader access, reach, message production, and evidence weight conceptually separable so that their effects can later be examined independently.

For the current design specification, see [Mass-Communication Opinion-Leader Model: Rough Design](docs/mass-communication-opinion-leader-rough-design.md).

## Quick Check

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -v