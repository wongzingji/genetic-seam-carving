## Algorithms
### Genetic Seam Carving
Seam Carving is in the discrete formulation, where connected paths of pixels across an image are proposed to be re-
moved to minimize distortions [Avidan and Shamir 2007]. Genetic algorithm is applied due to the large search space.

<img src="https://github.com/wongzingji/image_resizing/blob/master/images/visualize_seams.jpg" width="300" height="300">

#### Files
- gsc.py
  - Based on https://github.com/EvanLavender13/genetic-seam-carving/blob/master/carve.py
  - Reorganize the code, to modularize the following functions
    - The population in each generation (new_generation)
    - The seam to be carved in each step
  - Add in horizontal seam carving 

- multi_gsc.py
  - A new try of carving multiple pixels each row / column to make the algorithm non-greedy (i.e. global optimum)
  - Relax the sequential condition (while the result is not very good due to this)
  - Problem
    - Need to calculate #pixels carved each time to exactly reach the desired image size
    - There's glitch in the result since we loose the sequential condition

- operations.py
  - GA operators
    - Selection and variations (mutation & crossover) based on ) [Oliveira et al . 2015; Oliveira and Neto 2015]
    - Fitness function based on [Lavender 2019]

- vis.py
  - For visualization when performing the carving

#### TODO
- [ ] Try EMA-ES

### Image Warping

## Reference
[1] Saulo AF Oliveira, Francisco N Bezerra, and Ajalmar R Rocha Neto. 2015. Genetic seam carving: A genetic algorithm approach for content-aware image retargeting. In Iberian Conference on Pattern Recognition and Image Analysis. Springer, 700–707.

[2] Saulo AF Oliveira and Ajalmar R Rocha Neto. 2015. An improved genetic algorithms-based seam carving method. In 2015 Latin America Congress on Computational Intelligence (LA-CCI). IEEE, 1–6.

[3] Evan Lavender. 2019. genetic-seam-carving. https://github.com/EvanLavender13/genetic-seam-carving.

[4] Shai Avidan and Ariel Shamir. 2007. Seam carving for content-aware image resizing. In ACM SIGGRAPH 2007 papers. 10–es.
