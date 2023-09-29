# BYOL

[![Build Status](https://github.com/hv10/BYOL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/hv10/BYOL.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package aims to implement the self-supervised learning method from the paper ["Bootstrap Your Own Latent"](https://arxiv.org/abs/2006.07733) in Julias Flux ecosystem.

Extensions to the original paper are taken from [lucidrains/byol-pytorch](https://github.com/lucidrains/byol-pytorch/) and adapted to a more Julia like approach.

## Usage
The main byol method is implemented as a per batch update procedure.
For usage the to-be trained encoder needs to be "Wrapped" like so:

```julia
model = ... # some Flux Model
projector, predictor = make_mlp(32=>8), make_mlp(8=>8)
# note that predictor input & output dim has to match projector output dim!
online = WrappedNetwork(model, projector, predictor)
```

## ToDos

- [ ] Implement a few showcase use-cases
- [ ] Improve interface - current version with updater function is suboptimal