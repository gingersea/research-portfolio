# Post-run oracle terminology clarification

The frozen protocol calls the known-dynamics oracle an upper bound. More
precisely it is a **known-dynamics reference using the same controller**.
Knowing A/B does not guarantee globally optimal planning: this controller solves
an unconstrained finite-horizon quadratic objective and clips its applied action.
It is not a certified optimal constrained controller or bound over all policies.

The independent review identified this wording after the run. The original
protocol, code, raw data, controller and metrics remain unchanged. The canonical
Chinese synthesis and the generated English report use reference terminology.
