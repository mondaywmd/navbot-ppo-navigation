# V7 preparation results

- Fresh scene seed: `7042604`
- Scene manifest: 24 exact pairs / 48 scenes
- Unit tests: 5 passed
- Zero-update smoke: passed
- Final-V5 consistency loss on deterministic synthetic batch: `1.255619`
  (`linear=0.319327`, `angular=0.936292`)
- Actor state digest before and after backward:
  `4643c506d3651b8d2a685eccaef900e2bc2ec3dc26e0592a82a86657ade6af1f`
- Optimizer created: no
- Optimizer step called: no
- PPO started: no
- V7 checkpoint created: no

This verifies preparation and gradient plumbing only. It is not a training or
performance result, and no permanent OOD data was accessed.
