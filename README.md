# Opinion Model

## Opinion-Leader Branch

`opleader` implements the opinion-leader-only scenario of OLIM 2.0 after information about a focal issue has entered the social system. It isolates opinion-leader mechanisms without platform recommendation or network adaptation.

The current model contains opinion-leader and ordinary agents in a fixed directed information-access network. Every agent may originate a message, leaders receive an origination advantage, and leader messages receive a relative evidence multiplier. Originated messages travel only through configured ties, retain their source identity, and affect recipients only in the next synchronous update.

Current design documents:

- [Mass-Communication Opinion-Leader Model: Rough Design](docs/mass-communication-opinion-leader-rough-design.md)
- [Posting Origination](docs/posting-origination.md)
- [Message Selection](docs/message-selection.md)
- [Message Aggregation](docs/message-aggregation.md)
