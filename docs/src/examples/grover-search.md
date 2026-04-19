# Grover Search

> Amplify the marked item out of an unstructured database of \\( N \\) items
> in \\( O(\sqrt{N}) \\) oracle queries — two reflections compose to a
> rotation in a two-dimensional subspace and the marked amplitude grows.

## Background

Grover introduced his search algorithm in 1996[^grover]. The problem is
unstructured search: given oracle access to a Boolean function
\\( f : \{0, 1\}^n \to \{0, 1\} \\) with a single marked input \\( w \\)
satisfying \\( f(w) = 1 \\), locate \\( w \\). Classically no strategy does
better than \\( \Theta(N) \\) oracle queries on average, where
\\( N = 2^n \\); Grover's algorithm uses \\( O(\sqrt{N}) \\). This
quadratic speedup is modest compared to the exponential gap in
[Bernstein–Vazirani](./bernstein-vazirani.md), but it applies to a *generic*
black-box predicate rather than to a promised-structured function, so it
shows up as a subroutine far more often.

Applications include heuristic SAT solvers (Grover inside a branching
search tree), collision-finding speedups, and *amplitude estimation*, which
generalizes Grover by running phase estimation on the Grover iteration
operator to count the marked inputs without finding one. The geometric
picture — two reflections compose to a rotation in a two-dimensional
subspace spanned by the marked and unmarked components — is what makes the
algorithm memorable.

Nielsen and Chuang cover Grover's algorithm in §6.1[^nc].

## The math

Let \\( N = 2^n \\). The oracle takes two canonical forms. The *bit* oracle
\\( U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle \\) becomes, via
the usual ancilla-in-\\( |-\rangle \\) trick, the *phase* oracle

$$ O\,|x\rangle \;=\; (-1)^{f(x)}\,|x\rangle. $$

\\( O \\) flips the sign of \\( |w\rangle \\) and leaves every other
basis state alone.

**Amplitude-amplification geometry.** Define three states on the
\\( n \\)-qubit register:

- \\( |w\rangle \\) — the marked state (one basis vector).
- \\( |b\rangle = \tfrac{1}{\sqrt{N-1}}\sum_{x \ne w}|x\rangle \\) — the
  uniform superposition over the \\( N - 1 \\) unmarked states.
- \\( |s\rangle = H^{\otimes n}|0\rangle^{\otimes n} = \tfrac{1}{\sqrt{N}}\sum_{x}|x\rangle \\)
  — the uniform superposition the algorithm starts in.

Decomposing \\( |s\rangle \\) in the orthonormal basis
\\( \{|w\rangle, |b\rangle\} \\),

$$ |s\rangle \;=\; \sin\theta\,|w\rangle + \cos\theta\,|b\rangle, \qquad \sin\theta \;=\; \frac{1}{\sqrt{N}}. $$

**Grover iteration = two reflections.** The oracle \\( O \\) negates the
\\( |w\rangle \\) component and fixes the \\( |b\rangle \\) component, so
in the \\( (|w\rangle, |b\rangle) \\) plane it is the reflection across
\\( |b\rangle \\). The *diffusion* operator

$$ D \;=\; 2|s\rangle\langle s| - I \;=\; H^{\otimes n}\,(2|0\rangle\langle 0| - I)\,H^{\otimes n} $$

is the reflection across \\( |s\rangle \\). A composition of two reflections
with angle \\( \theta \\) between their axes is a rotation by \\( 2\theta \\)
in the plane they span. Hence

$$ G \;=\; D \cdot O $$

is a rotation by \\( 2\theta \\) in the \\( (|w\rangle, |b\rangle) \\)
plane, directed toward \\( |w\rangle \\).

**After \\( k \\) iterations** the state lies at angle \\( (2k+1)\theta \\)
from \\( |b\rangle \\):

$$ |\psi_k\rangle \;=\; G^k|s\rangle \;=\; \sin((2k+1)\theta)\,|w\rangle + \cos((2k+1)\theta)\,|b\rangle. $$

A measurement returns \\( w \\) with probability
\\( \sin^2((2k+1)\theta) \\). The optimal number of iterations satisfies
\\( (2k+1)\theta \approx \pi/2 \\), i.e.

$$ k_{\mathrm{opt}} \;\approx\; \frac{\pi}{4\theta} - \frac{1}{2} \;\approx\; \frac{\pi}{4}\sqrt{N}, $$

using \\( \theta \approx \sin\theta = 1/\sqrt{N} \\) for large \\( N \\).
That square-root is the Grover speedup.

**Concrete case.** With \\( n = 3 \\), \\( N = 8 \\), one marked state, we
have \\( \sin\theta = 1/\sqrt{8} \approx 0.3536 \\), so
\\( \theta \approx 0.3614 \\) rad (20.7°) and
\\( k_{\mathrm{opt}} \approx \pi/(4 \cdot 0.3614) - 1/2 \approx 1.67 \\).
Rounding up to \\( k = 2 \\), the predicted success probability is

$$ \sin^2(5\theta) \;=\; \sin^2(1.807) \;\approx\; 0.9454. $$

Note that \\( k = 3 \\) would *overshoot*: the state rotates past
\\( |w\rangle \\) and
\\( \sin^2(7\theta) = \sin^2(2.530) \approx 0.326 \\). Grover is periodic
— running the iteration forever does not help, the amplitude oscillates.

## The circuit

![Grover circuit](./generated/svg/grover-marked-5.svg)

Thirty-five elements on three qubits, for the marked basis index
\\( w = 5 \\). The circuit has three stages:

1. \\( H^{\otimes 3} \\) — three Hadamards preparing the uniform
   superposition \\( |s\rangle \\) from \\( |0\rangle^{\otimes 3} \\).
2. **Grover iteration 1**: oracle \\( O_w \\), then diffusion \\( D \\).
3. **Grover iteration 2**: identical oracle, identical diffusion.

**Oracle for marked state 5**. The integer 5 in binary, padded to three
bits, is \\( 101 \\). Under the yao-rs bit-ordering convention (qubit 0
at the MSB) this is \\( q_0 = 1,\ q_1 = 0,\ q_2 = 1 \\), so the marked
state is \\( |w\rangle = |101\rangle \\). The standard construction turns
\\( O_w \\) into a fixed "mark-the-all-ones-string" operation by sandwiching
it between \\( X \\) gates on every qubit whose marked bit is 0. In this
case that is qubit 1 only. The all-ones phase flip itself is a multi-
controlled \\( Z \\) with the last qubit as target and the remaining qubits
as controls — a CCZ. In short,

$$ O_w \;=\; X_1 \cdot \mathrm{CCZ}_{q_0, q_1 \to q_2} \cdot X_1. $$

**Diffusion**. The identity
\\( 2|s\rangle\langle s| - I = H^{\otimes n}(2|0\rangle\langle 0| - I)H^{\otimes n} \\)
lets us build \\( D \\) out of Hadamards and a mark-the-all-zeros phase
flip. The all-zeros phase flip is the all-ones phase flip sandwiched by
\\( X^{\otimes n} \\): every qubit's 0 and 1 swap, so

$$ D \;=\; H^{\otimes 3} \cdot X^{\otimes 3} \cdot \mathrm{CCZ}_{q_0, q_1 \to q_2} \cdot X^{\otimes 3} \cdot H^{\otimes 3}. $$

The CCZ is encoded as a `Z` gate on target `[2]` with `controls: [0, 1]`;
the following JSON excerpt shows the initial Hadamard layer and the full
first Grover iteration (17 of 35 gates). The second iteration is
bit-for-bit identical. The format follows the
[Circuit JSON Conventions](../conventions.md):

```json
{
  "num_qubits": 3,
  "elements": [
    {"type": "gate", "gate": "H", "targets": [0]},
    {"type": "gate", "gate": "H", "targets": [1]},
    {"type": "gate", "gate": "H", "targets": [2]},
    {"type": "gate", "gate": "X", "targets": [1]},
    {"type": "gate", "gate": "Z", "targets": [2], "controls": [0, 1]},
    {"type": "gate", "gate": "X", "targets": [1]},
    {"type": "gate", "gate": "H", "targets": [0]},
    {"type": "gate", "gate": "H", "targets": [1]},
    {"type": "gate", "gate": "H", "targets": [2]},
    {"type": "gate", "gate": "X", "targets": [0]},
    {"type": "gate", "gate": "X", "targets": [1]},
    {"type": "gate", "gate": "X", "targets": [2]},
    {"type": "gate", "gate": "Z", "targets": [2], "controls": [0, 1]},
    {"type": "gate", "gate": "X", "targets": [0]},
    {"type": "gate", "gate": "X", "targets": [1]},
    {"type": "gate", "gate": "X", "targets": [2]},
    {"type": "gate", "gate": "H", "targets": [0]}
  ]
}
```

[Full Grover JSON](./generated/circuits/grover-marked-5.json) (35 gates).

> **Bit ordering callout.** The marked integer 5 reads \\( 101_2 \\), which
> in the qubit-0-MSB convention means \\( q_0 = 1,\ q_1 = 0,\ q_2 = 1 \\).
> That matches the oracle's \\( X \\)-sandwich pattern: the \\( X \\) lands
> on qubit 1 precisely because \\( q_1 \\) is the only bit that is 0 in the
> marked state. See [bit ordering](../conventions.md#bit-ordering) for the
> full rule.

## Running it

Run from the repository root. This rebuilds the CLI, regenerates every
artifact embedded on this page via the bundled shell workflow, and
refreshes the plot:

```bash
cargo build -p yao-cli --no-default-features
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## Interpreting the result

![Grover probabilities](./generated/plots/grover-marked-5-probs.svg)

The probability array has `probabilities[5] = 0.9453` and every other
entry \\( \approx 0.0078 \\). Under the qubit-0-MSB convention, index 5
is \\( |q_0 q_1 q_2\rangle = |101\rangle \\) — the marked state. The
remaining weight, \\( 1 - 0.9453 \approx 0.0547 \\), is spread evenly
over the seven unmarked indices: \\( 0.0547 / 7 \approx 0.00781 \\) each,
matching the observed value.

Compare to theory. Predicted peak probability after \\( k = 2 \\)
iterations with \\( N = 8 \\) is

$$ \sin^2((2 \cdot 2 + 1)\theta) \;=\; \sin^2(5 \cdot 0.3614) \;=\; \sin^2(1.807) \;\approx\; 0.9454. $$

Measured 0.9453, predicted 0.9454 — agreement to five decimals, with the
residual one part in \\( 10^5 \\) attributable to float rounding in the
arcsin and sine evaluations.

Two observations. First, \\( 94.5\% \\) is not \\( 100\% \\) and cannot
be made so with this \\( N \\). The rotation angle \\( 2\theta \\) does
not divide \\( \pi \\) exactly, so no integer number of iterations lands
precisely on \\( |w\rangle \\); the closest approach at \\( k = 2 \\)
overshoots \\( \pi/2 \\) by a small amount (\\( 5\theta \approx 1.807 > \pi/2 \approx 1.571 \\)),
and the unmarked component \\( \cos(5\theta) \\) has size \\( \sim 0.23 \\).
For larger \\( N \\) the angle \\( \theta \\) shrinks and the residual
unmarked probability drops as \\( O(1/N) \\). Second, a third iteration
would overshoot further: at \\( k = 3 \\) the state lies at angle
\\( 7\theta \approx 2.530 \\), well past \\( \pi/2 \\), with
\\( \sin^2(7\theta) \approx 0.326 \\) — *worse* than one iteration. Grover
is not a "more is better" algorithm; the amplitude swings sinusoidally
in the \\( (|w\rangle, |b\rangle) \\) plane and the right stopping time
has to be computed in advance.

## Variations & next steps

- **Different marked states.** Rerun the script with any
  \\( \text{marked} \in \{0, 1, \dots, 7\} \\); the peak moves but the
  oracle's \\( X \\)-sandwich pattern adapts automatically to the new
  binary expansion, and the peak height stays \\( \approx 0.9453 \\).
- **Different register size.** Edit the script's \\( n \\) and the
  iteration count. The theoretical optimum is
  \\( \lfloor \pi / (4 \arcsin(1/\sqrt{N})) \rfloor \\); for \\( n = 4 \\)
  that is \\( \lfloor 3.14\ldots \rfloor = 3 \\), for \\( n = 6 \\) it is
  \\( 6 \\), and the peak probability approaches 1 as \\( N \\) grows.
- **Multiple marked states.** If \\( M \\) of the \\( N \\) inputs satisfy
  \\( f(x) = 1 \\), the rotation angle becomes
  \\( \theta = \arcsin(\sqrt{M/N}) \\) and the optimal number of iterations
  shrinks to \\( \approx \tfrac{\pi}{4}\sqrt{N/M} \\); this and the tight
  constants on the success probability are worked out by Boyer, Brassard,
  Høyer, and Tapp[^bbht]. Knowing \\( M \\) in advance matters; overshooting
  is easy.
- **Amplitude estimation.** Running phase estimation on the Grover
  iteration \\( G \\) recovers \\( \theta \\), hence \\( M / N \\), without
  ever identifying a specific marked input. See
  [Phase Estimation](./phase-estimation.md) for the readout block that
  slots in here.
- Compared to [Bernstein–Vazirani](./bernstein-vazirani.md) — which uses
  the same H–oracle–H scaffold for a single-shot deterministic readout —
  Grover's speedup is only quadratic. BV exploits a promise on \\( f \\)
  (it is linear over \\( \mathbb{Z}_2 \\)); Grover makes no such promise
  and so cannot beat \\( \sqrt{N} \\).
- **Deferred.** Variational Grover generators and structured oracles are
  tracked in issue #33.

## References

[^grover]: L. K. Grover, "A fast quantum mechanical algorithm for database
    search", in *Proc. 28th Annual ACM Symposium on Theory of Computing*
    (ACM, 1996), pp. 212–219; arXiv:quant-ph/9605043.

[^bbht]: M. Boyer, G. Brassard, P. Høyer, and A. Tapp, "Tight bounds on
    quantum searching", *Fortschr. Phys.* **46**, 493 (1998);
    arXiv:quant-ph/9605034.

[^nc]: M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
    Information*, 10th Anniversary Edition (Cambridge University Press,
    2010), §6.1 (the quantum search algorithm).
