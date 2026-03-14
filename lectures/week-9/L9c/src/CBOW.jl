"""
    build_cbow_pairs(sentences::Array{String,1}, vocabulary::Dict{String, Int64};
        window_size::Int64 = 2) -> Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}

Generate CBOW training pairs from a corpus. For each word position in each sentence,
the context is the sum of one-hot vectors of surrounding words within the window,
and the target is the one-hot vector of the center word.

### Arguments
- `sentences::Array{String,1}`: array of sentence strings.
- `vocabulary::Dict{String, Int64}`: dictionary mapping words to vocabulary indices.
- `window_size::Int64`: number of tokens on each side to include in context (default: 2).

### Returns
- `Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}`: array of (context vector, target one-hot) tuples.
"""
function build_cbow_pairs(sentences::Array{String,1}, vocabulary::Dict{String, Int64};
    window_size::Int64 = 2)::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}

    # initialize -
    vocab_size = length(vocabulary);
    pairs = Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}();

    # generate training pairs -
    for sentence in sentences
        augmented_sentence = "<bos> " * sentence * " <eos>";
        words = split(lowercase(augmented_sentence)) .|> String;
        word_indices = [get(vocabulary, word, vocabulary["<unk>"]) for word in words];

        for i ∈ eachindex(word_indices)
            target_index = word_indices[i];
            target_onehot = zeros(Float64, vocab_size);
            target_onehot[target_index] = 1.0;

            context_vector = zeros(Float64, vocab_size);
            left = max(1, i - window_size);
            right = min(length(word_indices), i + window_size);
            for j ∈ left:right
                if j == i
                    continue
                end
                context_vector[word_indices[j]] += 1.0;
            end

            push!(pairs, (context_vector, target_onehot));
        end
    end

    return pairs;
end

"""
    train_cbow(training_pairs::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}, vocab_size::Int64;
        d_h::Int64 = 5, η::Float64 = 0.01, num_epochs::Int64 = 500) -> Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}

Train a CBOW model using gradient descent with full softmax.
The forward pass computes h = W₁x, u = W₂h, ŷ = softmax(u).
Gradients are computed manually and weights are updated in-place.

### Arguments
- `training_pairs::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}`: array of (context, target) pairs.
- `vocab_size::Int64`: size of the vocabulary.
- `d_h::Int64`: embedding dimension (default: 5).
- `η::Float64`: learning rate (default: 0.01).
- `num_epochs::Int64`: number of training epochs (default: 500).

### Returns
- `Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}`: tuple of (W₁, W₂, loss_history).
"""
function train_cbow(training_pairs::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}, vocab_size::Int64;
    d_h::Int64 = 5, η::Float64 = 0.01, num_epochs::Int64 = 500)::Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}

    # initialize weight matrices -
    W₁ = randn(d_h, vocab_size) * 0.1;
    W₂ = randn(vocab_size, d_h) * 0.1;

    # training loop -
    loss_history = zeros(num_epochs);
    for epoch ∈ 1:num_epochs
        epoch_loss = 0.0;
        for (x, y) in training_pairs

            # forward pass -
            h = W₁ * x;
            u = W₂ * h;
            ŷ = softmax(u);

            # cross-entropy loss -
            loss = -sum(y .* log.(ŷ .+ 1e-12));
            epoch_loss += loss;

            # backward pass -
            δ_u = ŷ - y;
            ∂L_∂W₂ = δ_u * h';
            δ_h = W₂' * δ_u;
            ∂L_∂W₁ = δ_h * x';

            # gradient descent update -
            W₁ .-= η * ∂L_∂W₁;
            W₂ .-= η * ∂L_∂W₂;
        end
        loss_history[epoch] = epoch_loss / length(training_pairs);
    end

    return (W₁, W₂, loss_history);
end
