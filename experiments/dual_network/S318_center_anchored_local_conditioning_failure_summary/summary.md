# S318 local-conditioning failure summary

S313-S317 showed that simple local-shape conditioning is not enough. The matched-split `current_reference` reproduced val/test IoU `0.037245` / `0.072368` with zero-IoU `8/10` / `9/10`. `conditioning_center_bin` improved train fit to `0.999749` / `0.992481`, but held-out val/test IoU fell to `0.027215` / `0.067059` and zero-IoU stayed high at `9/10` / `9/10`.

This stage therefore treats the failure as a causal diagnosis problem: does final mask failure come mainly from center decode, or from local-shape prediction after center is fixed?

Boundaries: no multi-seed, no extra steps, no y-loss sweep, no bound sweep, no local-conditioning sweep, and no new COMSOL data. The S185/S181 branch candidate remains unchanged.
