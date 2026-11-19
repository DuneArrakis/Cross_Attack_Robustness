# Empirical network inputs

Each file is a single undirected, unweighted topology. The header records the number of vertices. Each remaining line contains two zero-based vertex indices and an empty edge-attribute dictionary. The file hashes below match the seven `Original` rows in the corresponding archive's `evaluations.csv`.

| Setting | Input topology | Nodes | Edges | SHA-256 |
| --- | --- | ---: | ---: | --- |
| EUpower217 | [graph_EUpower.txt](graph_EUpower.txt) | 217 | 320 | `5972f2df6416433dd3da93bbb1199724d1f5e1fb042722efa5cc20bcf97c49ae` |
| USAir97 | [graph_USAir97_0.txt](graph_USAir97_0.txt) | 332 | 2,126 | `3e0fdd52410d2e1b28ad93439f9b450fffb4b320a4e8be0068e7fee65d5bff92` |

**EUpower217.** The western European subgraph was reconstructed from the 2002 winter off-peak case of the Zhou–Bialek model. All vertices in the selected region (model areas 1 and 2) were retained. Internal connections were converted to undirected edges and parallel edges were merged. The resulting graph is connected. Its node and edge counts, assortativity, mean path length and clustering agree with the values reported by Zeng and Liu at their published precision. Earlier commits contain a 213-node, 316-edge file under the same filename. Table 2 uses the 217-node graph linked above. [Zhou and Bialek (2005)](https://doi.org/10.1109/TPWRS.2005.846178) · [Zeng and Liu (2012)](https://doi.org/10.1103/PhysRevE.85.066130)

**USAir97.** The 332 vertices represent airports and the edges represent routes. The file preserves the vertex indices and edge order used for the published evaluations. [Network Data Repository](https://networkrepository.com/)

Run `python reproduce.py` in the study container to check both files against the measurements and rebuild Table 2.
