using MLUtils
using MLDatasets
using Metalhead
using Augmentor
using Images
using Optimisers
using ProgressBars
using Flux
using StatsPlots
using Random
using Distributions
using BYOL

plotlyjs()

"""
This extra mapping function is neccessary to support batched inputs.
It also makes the application of an augmentation deterministic under seed `s`.
(Which is neccessary for BYOL to learn over multiple epochs.)
"""
aug(x, pipeline; s=1234) = begin
    Random.seed!(s)
    mapslices(x -> augment(x, pipeline), x, dims=(1, 2))
end

"""
Run BYOL as an example self-supervision training method on the MNIST dataset.
"""
main(emb_dim=2) = begin
    # Load the MNIST dataset
    train = MNIST(split=:train)
    idxs = sample(axes(train.features, 3), 5000, replace=false) # for speed up
    X_train = train.features[:, :, idxs]
    y_train = train.targets[idxs]

    # define the augmentations used to differentiate samples
    # feel free to play around with this and see how it affects the results
    # make sure that your augmentations are invariant under the downstream task
    #   i.e. MNIST labels are not invariant under flipping operations: FlipY(6) ≈ 9
    pl = ElasticDistortion(6, scale=0.3, border=true) |>
         Rotate([10, -5, -3, 0, 3, 5, 10]) |>
         ShearX(-10:10) * ShearY(-10:10) |>
         CropSize(28, 28) |>
         Zoom(0.9:0.1:1.2)

    # Create the online and target network
    # The online network is the network that is trained.
    # The target network is the network that is used to create the target for the online network.
    # For ease we use a ResNet16:
    model = Chain(
        MLUtils.flatten,
        Dense(28^2, 8, relu),
        Dense(8, 16, relu),
        Dense(16, emb_dim, tanh_fast)
    )
    projector, predictor = make_mlp(emb_dim => 8), make_mlp(8 => 8)
    online = WrappedNetwork(model, projector, predictor)
    target = deepcopy(online)
    opt_state = Optimisers.setup(Optimisers.Adam(), online)

    train_dl = MLUtils.DataLoader((X_train, y_train), batchsize=256, shuffle=true, collate=true)

    figs = []
    for i in 1:50
        @info "Epoch $i"
        for (i, (x, _)) in ProgressBar(enumerate(train_dl))
            opt_state, online, target = byol_update!(
                opt_state, x, online, target,
                x -> aug(x, pl; s=i + 1), x -> aug(x, pl; s=i)
            )
        end
        push!(figs, plot_mnist_emb_space(online.net))
        display(figs[end])
    end
    anim = @animate for fig in figs
        plot(fig)
    end
    gif(anim, joinpath(@__DIR__, "out.gif"))
    return online, figs
end

plot_mnist_emb_space(model) = begin
    Random.seed!(1234)
    test = MNIST(split=:test)
    idxs = sample(axes(test.features, 3), 1000, replace=false)
    X_test = test.features[:, :, idxs]
    y_test = test.targets[idxs]
    yhat = model(X_test)
    if ndims(yhat) == 3
        scatter(yhat[1, :], yhat[2, :], yhat[3, :], color=y_test, legend=false, markersize=0.9, lims=(-1, 1))
    else
        scatter(yhat[1, :], yhat[2, :], color=y_test, legend=false, lims=(-1, 1))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end