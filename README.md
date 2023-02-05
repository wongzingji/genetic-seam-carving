#### Genetic seam carving

##### algorithm

GA



##### based on

https://github.com/EvanLavender13/genetic-seam-carving/blob/master/carve.py



##### modification

- 重新组织了下代码，把以下功能模块化：
  - determine the population in each generation (`new_generation`)
  - determine the seams need to be carved in each step

- 原来的代码只包含vertical seam，加入了horizontal seam carving



##### TODO

- multi-seam carving

  改了一半，代码执行起来还有问题（目前hyperparameter中只能取`n=1`），原因见`problem`

- 看其他文献修改GA中的设定

- 看能不能利用现成的库，eg. DEAP, CMA-ES



##### problem

- multi-seam carving
  - each step carve多条，也许不能exactly达到想要的image size
  - 选择的多条seam可能会有pixel重合
  - 效果也许不好 -- once at a time是取每次最好的，multi是取每次top n好的

