# Team puffle - US/UK PETs Prize Challenge

This repo contains the submission code of [team puffle](https://drivendata.co/blog/federated-learning-pets-prize-winners-phases-2-3#puffle) at the [US/UK PETs Prize Challenge](https://petsprizechallenge.drivendata.org/). The implemention is written in Python with JAX and Haiku for model training.


## Code guide

See [`code_documentation.md`](code_documentation.md) which describes the code structure and provides a high-level overview of the implementation.

## Solution documentation

The full solution documentation is provided [here](https://hackmd.io/@kzl6/pets-challenge). Please note that this is an evolving document, and we will be making regular updates to this blog post to provide more discussions on privacy implementation, additional analyses, and open challenges. Additionally, keep an eye out for a potential updated version of this post on the [ML@CMU blog](https://blog.ml.cmu.edu/).

See also [our NeurIPS'22 paper](https://arxiv.org/abs/2206.07902) and [code](https://github.com/kenziyuliu/private-cross-silo-fl) based on which we developed our entry to the challenge.

## Dependencies

The dependencies are specified in [`runtime/environment-cpu.yml`](runtime/environment-cpu.yml) and [`runtime/environment-gpu.yml`](runtime/environment-gpu.yml) . Note that these files contain the dependencies sufficient for our local testing, and the official runtime during the challenge had other dependencies (e.g. see official CPU dependencies [here](https://github.com/drivendataorg/pets-prize-challenge-runtime/blob/main/runtime/environment-cpu.yml)).

## Team

See our team profile [here](https://drivendata.co/blog/federated-learning-pets-prize-winners-phases-2-3#puffle).

## Contact

For any questions, feel free to contact Ken Liu <kenziyuliu@gmail.com>.

## Citation

Please consider citing our paper if you find this repo useful:

```BibTeX
@article{liu2022privacy,
  title={On privacy and personalization in cross-silo federated learning},
  author={Liu, Ken and Hu, Shengyuan and Wu, Steven Z and Smith, Virginia},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={5925--5940},
  year={2022}
}
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
