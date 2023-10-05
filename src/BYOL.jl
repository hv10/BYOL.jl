module BYOL
#= 
This repository is a Python to Julia translation of the BYOL paper
extended using the code from lucidrains/byol-pytorch/
with some Julia-fication
=#

using Flux
using LinearAlgebra
using Statistics

export make_mlp, make_simsiammlp
export WrappedNetwork
export byol_update!

make_mlp(in_out::Pair; hidden_size::Int=4096) = Chain(
    Dense(in_out.first => hidden_size),
    BatchNorm(hidden_size),
    relu,
    Dense(hidden_size => in_out.second)
)

make_simsiammlp(in_out::Pair; hidden_size::Int=4096) = Chain(
    Dense(in_out.first => hidden_size; bias=false),
    BatchNorm(hidden_size),
    relu,
    Dense(hidden_size => hidden_size; bias=false),
    BatchNorm(hidden_size),
    relu,
    Dense(hidden_size => in_out.second; bias=false),
    BatchNorm(in_out.second; affine=False)
)

ema_update(old, new; beta=0.99) = begin
    ps, reconstruct = Flux.destructure(old)
    ps_new, _ = Flux.destructure(new)
    ps_upd = @. beta * ps + (1 - beta) * ps_new
    reconstruct(ps_upd)
end

"""
A wrapper for the online and target network of BYOL.
It connects the `net` to the `projector` and `predictor`.
Make sure that the predictor input and output size matches the projector output size.
Make sure that the projectors input shape is compatible with the wrapped networks output shape.
"""
struct WrappedNetwork
    net::Chain # fθ
    projector::Chain # gθ
    predictor::Chain # pθ
end
Flux.@functor WrappedNetwork

"""
Applies the online network to the input `x` and returns the output of the wrapped network, the projector and the predictor.
"""
(m::WrappedNetwork)(x) = begin
    hx = m.net(x)
    px = m.projector(hx)
    qx = m.predictor(px)
    return hx, px, qx
end

"""
An almost literal translation of the BYOL papers algorithm.
It performs the byol update routine on the online and target network for one batch of data.

Mapping the concepts from the paper to the code:
online = fθ, target = fξ, 
gθ, pθ, gξ are rolled into online/target network
aug1 = t, aug2 = t′

Inputs:
- opt_state: the result of Optimisers.setup on the online network
- x: the batch of data
- online::WrappedNetwork: the online network
- target::WrappedNetwork: the target network
- aug1::Function: the first augmentation function
- aug2::Function: the second augmentation function

Returns:
- opt_state: the updated optimiser state
- online::WrappedNetwork: the updated online network
- target::WrappedNetwork: the updated target network
"""
byol_update!(
    opt_state, x,
    online::WrappedNetwork, target::WrappedNetwork,
    aug1::Function, aug2::Function;
    use_momentum=true, beta=0.99
) = begin
    a1 = aug1(x)
    a2 = aug2(x)
    grad = Flux.gradient(online) do model
        # for every element in batch (last axis = batch axis)
        # apply aug1 and aug2 to batch
        # apply encoder model to augmented batch
        # apply projector to encoder output
        _, _, q1 = model(a1)
        _, _, q2 = model(a2)
        # notice the flipped augmentation for target
        # notice that we do not apply the predictor to target we are only interested in the projected version
        # notice that target is not part of the gradient
        _, zt1, _ = target(a2)
        _, zt2, _ = target(a1)
        # normalize the vectors where neccessary
        # calc loss acc. to paper
        loss = 2 - 2 * (
            (dot(q1, zt1) / (norm(q1) * norm(zt1))) +
            (dot(q2, zt2) / (norm(q2) * norm(zt2)))
        )
        return mean(loss)
    end
    # apply optimizer
    Flux.update!(opt_state, online, grad[1])
    # update target network / allow for momentum free version
    if use_momentum
        target = ema_update(target, online; beta=beta)
    else
        target = deepcopy(online)
    end
    return opt_state, online, target
end

end
