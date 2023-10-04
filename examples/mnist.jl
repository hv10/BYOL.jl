using MLUtils
using MLDatasets
using Metalhead
using Augmentor
using Images
using Optimisers
using ProgressBars
using Flux
using BYOL


main() = begin
    # Load the MNIST dataset
    train = MNIST(split=:train)
    test = MNIST(split=:test)
    X_train = train.features
    y_train = train.targets
    X_test = test.features
    y_test = test.targets


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
        Dense(16, 10),
        softmax
    )
    projector, predictor = make_mlp(10 => 16), make_mlp(16 => 16)
    online = WrappedNetwork(model, projector, predictor)
    target = deepcopy(online)
    opt_state = Optimisers.setup(Optimisers.Adam(), online)

    aug(x) = mapslices(x -> augment(x, pl), x, dims=(1, 2))


    train_dl = MLUtils.DataLoader((X_train, y_train), batchsize=2, shuffle=true, collate=true)

    for i in 1:20
        @info "Epoch $i"
        for (x, y) in ProgressBar(train_dl)
            @show size(online(aug(x))[3])
            opt_state, online, target = byol_update!(opt_state, x, online, target, aug, aug)
            break
        end
        break
    end
    return online
end