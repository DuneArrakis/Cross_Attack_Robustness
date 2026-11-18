## Attack Methods and References

This repository includes or wraps the following attack methods:

- HDA: high-degree adaptive attack baseline implemented in `HDA.py`
- CI: Collective Influence attack, adapted with reference to
  [zhfkt/ComplexCi](https://github.com/zhfkt/ComplexCi)
- CoreHD: CoreHD attack, adapted with reference to
  [hcmidt/corehd](https://github.com/hcmidt/corehd)
- FINDER: deep reinforcement learning based attack, wrapped from `FINDER_ND/`
  with reference to [FFrankyy/FINDER](https://github.com/FFrankyy/FINDER)
- EI: Explosive Immunization attack, adapted with reference to
  [pclus/explosive-immunization](https://github.com/pclus/explosive-immunization)

- SRA applies the local degree and neighbor-degree formula from Zheng et al. to the residual graph. [Discovering network resilience formulas with symbolized reinforcement learning](https://doi.org/10.1038/s41467-026-78269-3).
- DomiRank uses a fixed order within each removal sequence. [DomiRank Centrality reveals structural fragility of complex networks via node dominance](https://doi.org/10.1038/s41467-023-44257-0).

Please also check the licenses and citation requirements of the referenced
upstream repositories before redistribution.
