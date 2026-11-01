# KNC-0 Phase 4A results

## Setup

The full CPU matrix was run from commit
`aa36c9a9af85f4e99bdfff1b0db53434094f9d64`: four input dimensions, six hidden
widths, 30 independent CSPRNG examples per MLP configuration, and 30 linear
baselines per input dimension. This produced 840 runs. Candidate outputs were
fixed without target access and scored afterward.

## Memorization

Every model, including all 120 linear baselines, exactly memorized its single
12-bit token after rounding: 840/840 successes and zero convergence failures.
Mean stopping points across configurations ranged from 138.3 to 250 training
steps. Consequently the descriptive 95%-success transition was already at the
smallest tested width, `h=1`, for every input dimension. Phase 4A found no
non-trivial width transition in the tested grid.

This result is structurally important: memorizing one scalar association does
not require a wide or deep network. Parameter count ranged from 2 for the
smallest linear baseline to 20,673 for `d=256, h=64`, without changing exact
memorization success.

## Outputs away from the memorized input

Exact memorization did not imply uniform off-key behavior. Mean random-input
candidate error generally increased with input dimension and often with width,
but was non-monotonic. For `d=1`, error ranged from 84 tokens (`h=2`) to 464
(`h=64`). For `d=256`, it ranged from 211 (`h=1`) to 1,494 (`h=8`), then fell to
1,180 at `h=64`.

Linear random-input error was much larger: 1,184 tokens at `d=1`, 1,581 at
`d=4`, 2,136 at `d=16`, and 1,892 at `d=256`. Thus exact training success alone
does not predict the numerical output distribution, and neither width nor total
parameter count supports a simple monotonic law here.

## Scaling and limitations

Mean training runtime ranged from 0.058 to 0.178 seconds per run. Serialized size
tracks parameter count as expected; plots use a logarithmic parameter-count axis.
The experiment concerns one random 12-bit scalar and characterizes learning and
off-key numerical behavior. It does not measure practical secrecy, multi-token
capacity, semantic memory, or resistance to extraction.

`aggregate.json` contains Wilson intervals for exact/proportion metrics and mean
intervals for continuous metrics. `per_run.csv` retains all 840 runs. Five SVG
plots separate width and input-dimension effects and show error, runtime and model
size against parameter count. Review is required before extending to multiple
tokens.
