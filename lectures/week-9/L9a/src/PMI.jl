"""
    build_cooccurrence_matrix(sentences::Array{String,1}, vocabulary::Dict{String, Int64};
        window_size::Int64 = 2) -> Array{Int64, 2}

Build a word-context co-occurrence matrix from a corpus. For each word in each sentence,
counts how many times each other word appears within `window_size` positions on either side.

### Arguments
- `sentences::Array{String,1}`: array of sentence strings.
- `vocabulary::Dict{String, Int64}`: dictionary mapping words to vocabulary indices.
- `window_size::Int64`: number of tokens on each side to consider as context (default: 2).

### Returns
- `Array{Int64, 2}`: co-occurrence matrix of shape `(vocab_size, vocab_size)`.
"""
function build_cooccurrence_matrix(sentences::Array{String,1}, vocabulary::Dict{String, Int64};
    window_size::Int64 = 2)::Array{Int64, 2}

    # initialize -
    vocab_size = length(vocabulary);
    cooccurrence_matrix = zeros(Int64, vocab_size, vocab_size);

    # populate the co-occurrence matrix -
    for sentence in sentences
        augmented_sentence = "<bos> " * sentence * " <eos>";
        words = split(lowercase(augmented_sentence)) .|> String;
        word_indices = [get(vocabulary, word, vocabulary["<unk>"]) for word in words];

        for i ∈ eachindex(word_indices)
            target_index = word_indices[i];
            left = max(1, i - window_size);
            right = min(length(word_indices), i + window_size);
            for j ∈ left:right
                if j == i
                    continue
                end
                context_index = word_indices[j];
                cooccurrence_matrix[target_index, context_index] += 1;
            end
        end
    end

    return cooccurrence_matrix;
end

"""
    build_pmi_matrices(cooccurrence_matrix::Array{Int64, 2}) -> Tuple{Array{Float64, 2}, Array{Float64, 2}}

Compute Pointwise Mutual Information (PMI) and Positive PMI (PPMI) matrices from a co-occurrence matrix.
PMI values are computed as `log2(P(w,c) / (P(w) * P(c)))`. Entries where any probability is zero are set to `-Inf`.
PPMI replaces negative PMI values with zero.

### Arguments
- `cooccurrence_matrix::Array{Int64, 2}`: co-occurrence matrix of shape `(vocab_size, vocab_size)`.

### Returns
- `Tuple{Array{Float64, 2}, Array{Float64, 2}}`: tuple of (PMI matrix, PPMI matrix), each of shape `(vocab_size, vocab_size)`.
"""
function build_pmi_matrices(cooccurrence_matrix::Array{Int64, 2})::Tuple{Array{Float64, 2}, Array{Float64, 2}}

    # initialize -
    vocab_size = size(cooccurrence_matrix, 1);
    total_pairs = sum(cooccurrence_matrix);

    # compute probabilities -
    P_wc = cooccurrence_matrix / total_pairs;
    P_w = vec(sum(P_wc, dims=2));
    P_c = vec(sum(P_wc, dims=1));

    # compute PMI -
    PMI_matrix = fill(-Inf, vocab_size, vocab_size);
    for i in 1:vocab_size
        for j in 1:vocab_size
            p_wc = P_wc[i, j];
            p_w = P_w[i];
            p_c = P_c[j];
            if p_wc == 0 || p_w == 0 || p_c == 0
                continue
            end
            PMI_matrix[i, j] = log2(p_wc / (p_w * p_c));
        end
    end

    # compute PPMI -
    PPMI_matrix = max.(PMI_matrix, 0.0);

    return (PMI_matrix, PPMI_matrix);
end
