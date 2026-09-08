# First circuit

Prepare an entangled pair, draw the circuit, and measure its output.
This walkthrough uses the [installed CLI](installation.md).

## Create a Bell circuit

```bash
yao example bell > bell.json
```

The file describes two operations: a Hadamard on qubit 0, followed by an X on
qubit 1 controlled by qubit 0. Both qubits start in \\( |0\rangle \\) when simulated.

## Draw it

```bash
yao visualize bell.json --output bell.svg
```

Open `bell.svg` in a browser:

![Bell circuit: H on qubit 0 followed by a controlled X on qubit 1.](examples/generated/svg/bell.svg)

## Compute probabilities

```bash
yao simulate bell.json | yao probs - --json
```

`simulate` produces a binary state vector. The pipe passes it to `probs`,
which computes a probability for each basis state. `-` means read from stdin.
The probabilities are approximately:

```json
{"num_qubits": 2, "locs": null, "probabilities": [0.5, 0.0, 0.0, 0.5]}
```

The pair is equally likely to be measured as \\( |00\rangle \\) or
\\( |11\rangle \\). The two qubits always agree.

## Take measurement samples

```bash
yao run bell.json --shots 1024
```

The terminal shows counts for `00` and `11`. Each should be near half the
shots, with variation between runs.

You've now created, rendered, and simulated the same circuit. Continue with
[simulation and measurement](states.md) to compute expectation values, or
[entangled states](examples/entangled-states.md) for the physics behind this example.
