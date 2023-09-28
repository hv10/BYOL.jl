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

struct WrappedNetwork
    net::Chain
    projector::Chain
    predictor::Chain
end
Flux.@functor WrappedNetwork
(m::WrappedNetwork)(x) = begin
    hx = m.net(x)
    px = m.projector(hx)
    qx = m.predictor(px)
    return hx, px, qx
end

# a literal translation of the papers algoithm
byol_update!(opt_state, x, online::WrappedNetwork, target::WrappedNetwork, aug1::Function, aug2::Function) = begin
    # online = fθ, target = fξ, projector, target_projector, predictor are rolled into online/target 
    # aug1 = t, aug2 = t′
    # take gradient of loss w.r.t model
    xdims = ndims(x)
    @show size(x)
    grad = Flux.gradient(online) do model
        # for every element in batch (last axis = batch axis)
        # apply aug1 and aug2 to batch
        # apply encoder model to augmented batch
        # apply projector to encoder output
        _, _, q1 = model(aug1(x))
        _, _, q2 = model(aug2(x))
        # notice the flipped augmentation for target
        # notice that we do not apply the predictor to target we are only interested in the projected version
        # notice the ignored gradient for target
        _, zt1, _ = target(aug2(x))
        _, zt2, _ = target(aug1(x))
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
    # update target network
    target = ema_update(target, online; beta=0.99)
    return opt_state, online, target
end

end
