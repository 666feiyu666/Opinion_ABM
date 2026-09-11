# Opinion Model

From Two-Step Flow to Algorithmically Mediated Influence: How Online Platforms
Transform Opinion Leadership in Opinion Dynamics.

This repository provides the shared implementation for comparing three
information environments:

- `null`: the control model with homogeneous message
  origination, selection, aggregation, and a static network;
- `opleader`: interpersonal diffusion with opinion-leader origination and
  source-weight advantages;
- `platform`: uniform platform-mediated reach beyond existing ties and
  exposure-driven adaptation of the directed following network.

`main` is the canonical integrated implementation. Case branches retain their
experiment-specific notebooks, configurations, figures, and exploratory work.
The historical thesis implementation and its artifacts remain on `legacy`.

## Mechanism Documentation

Opinion-leader case:

- [Rough design](docs/mass-communication-opinion-leader-rough-design.md)
- [Posting origination](docs/posting-origination.md)
- [Message selection](docs/message-selection.md)
- [Message aggregation](docs/message-aggregation.md)

Platform case:

- [Rough design](docs/platform-mediated-opinion-dynamics-rough-design.md)
- [Message selection](docs/platform-message-selection.md)
- [Network update](docs/platform-network-update.md)

Full framework:

- [Matched four-scenario comparison](docs/full-model-comparison.md)