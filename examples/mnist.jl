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


main() = begin
    # Load the MNIST dataset
    train = MNIST(split=:train)
    X_train = train.features
    y_train = train.targets


    # define the augmentations used to differentiate samples
    pl = Either(1 => FlipX(), 1 => FlipY(), 2 => NoOp()) |>
         Rotate(0:360) |>
         Resize(28, 28)

    # Create the online and target network
    # The online network is the network that is trained.
    # The target network is the network that is used to create the target for the online network.
    # For ease we use a ResNet16:
    model = Chain(
        MLUtils.flatten,
        Dense(28^2, 8, relu),
        Dense(8, 16, relu),
        Dense(16, 3)
    )
    projector, predictor = make_mlp(3 => 8), make_mlp(8 => 8)
    online = WrappedNetwork(model, projector, predictor)
    target = deepcopy(online)
    opt_state = Optimisers.setup(Optimisers.Adam(), online)

    aug(x) = mapslices(x -> augment(x, pl), x, dims=(1, 2))


    train_dl = MLUtils.DataLoader((X_train, y_train), batchsize=256, shuffle=true, collate=true)

    figs = []
    for i in 1:20
        @info "Epoch $i"
        for (x, _) in ProgressBar(train_dl)
            opt_state, online, target = byol_update!(opt_state, x, online, target, aug, aug)
        end
        push!(figs, plot_mnist_emb_space(online.net))
        display(figs[end])
    end
    @gif for fig in figs
        fig
    end fps = 4
    return online, figs
end

plot_mnist_emb_space(model) = begin
    test = MNIST(split=:test)
    idxs = sample(axes(test.features, 3), 100)
    X_test = test.features[:, :, idxs]
    y_test = test.targets[idxs]
    yhat = model(X_test)
    scatter(yhat[1, :], yhat[2, :], yhat[3, :], color=y_test, legend=false)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end