## Algorithms
### Genetic Seam Carving (discrete method)
Seam Carving is in the discrete formulation, where connected paths of pixels across an image are proposed to be removed to minimize distortions [Avidan and Shamir 2007]. Genetic algorithm is applied due to the large search space.

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

### Image Warping (continuous method)
A implementation of Mesh warping based on [7].

Objective: To find an optimal vertex positions $V^∗$ to warp the image in order to preserve the important quads as much as possible and allow the distortion of the unimportant quads.

Formulation: Formed as a constraint optimation problem, where the objective function is the shape distortion and bending of lines with respected to the vertex positions (the weight beween two items can be calculated by the energy map), and the constraint is imposed by ensuring the vertices not to be outside of the borders. By relaxing other parameters and only optimizing the vertex positions, the problem can be reduced to a series of linear problems. The initial quads are set by a simple scale, and the quad positions are optimized iteratively until convergence.

### Energy functions
To represent the importance.
- Forward energy [5] \
  Considers the effect on the retargeted image; \
  Shown to produces less discontinuities compared to the backward energy.
- Image gradiant
- Saliency map [6]
- Combination of forward energy & saliency map \
  Inspired by Wang et al. (2008) [7] where a combination of image gradient and saliency map is used.

## Reference
[1] Saulo AF Oliveira, Francisco N Bezerra, and Ajalmar R Rocha Neto. 2015. Genetic seam carving: A genetic algorithm approach for content-aware image retargeting. In Iberian Conference on Pattern Recognition and Image Analysis. Springer, 700–707. \
[2] Saulo AF Oliveira and Ajalmar R Rocha Neto. 2015. An improved genetic algorithms-based seam carving method. In 2015 Latin America Congress on Computational Intelligence (LA-CCI). IEEE, 1–6. \
[3] Evan Lavender. 2019. genetic-seam-carving. https://github.com/EvanLavender13/genetic-seam-carving. \
[4] Shai Avidan and Ariel Shamir. 2007. Seam carving for content-aware image resizing. In ACM SIGGRAPH 2007 papers. 10–es. \
[5] Michael Rubinstein, Ariel Shamir, and Shai Avidan. 2008. Improved seam carving for video retargeting. ACM transactions on graphics (TOG) 27, 3 (2008), 1–9. \
[6] Laurent Itti, Christof Koch, and Ernst Niebur. 1998. A model of saliency-based visual attention for rapid scene analysis. IEEE Transactions on pattern analysis and machine intelligence 20, 11 (1998), 1254–1259. \
[7] Yu-Shuen Wang, Chiew-Lan Tai, Olga Sorkine, and Tong-Yee Lee. 2008. Optimized scale-and-stretch for image resizing. In ACM SIGGRAPH Asia 2008 papers. 1–8.
