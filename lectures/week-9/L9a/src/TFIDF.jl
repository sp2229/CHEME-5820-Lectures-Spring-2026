"""
    build_tf_matrix(bow_matrix::Array{Int64, 2}) -> Array{Float64, 2}

Compute the term frequency (TF) matrix from a Bag of Words matrix.
Each entry is the raw word count normalized by the total number of words in that sentence.

### Arguments
- `bow_matrix::Array{Int64, 2}`: BoW matrix of shape `(num_sentences, vocab_size)`.

### Returns
- `Array{Float64, 2}`: TF matrix of shape `(num_sentences, vocab_size)`.
"""
function build_tf_matrix(bow_matrix::Array{Int64, 2})::Array{Float64, 2}

    # initialize -
    num_sentences = size(bow_matrix, 1);
    vocab_size = size(bow_matrix, 2);
    TF_matrix = zeros(Float64, num_sentences, vocab_size);

    # populate the TF matrix -
    for i in 1:num_sentences
        total_terms = sum(bow_matrix[i, :]);
        if total_terms == 0
            continue
        end
        for j in 1:vocab_size
            TF_matrix[i, j] = bow_matrix[i, j] / total_terms;
        end
    end

    return TF_matrix;
end

"""
    build_idf_dictionary(bow_matrix::Array{Int64, 2}, vocabulary::Dict{String, Int64},
        num_sentences::Int64) -> Dict{String, Float64}

Compute smoothed inverse document frequency (IDF) values for each term in the vocabulary.
Uses the smoothed formula: `idf(t) = ln((N + 1) / (df(t) + 1))` to avoid division by zero.

### Arguments
- `bow_matrix::Array{Int64, 2}`: BoW matrix of shape `(num_sentences, vocab_size)`.
- `vocabulary::Dict{String, Int64}`: dictionary mapping words to vocabulary indices.
- `num_sentences::Int64`: total number of documents in the corpus.

### Returns
- `Dict{String, Float64}`: dictionary mapping words to their IDF scores.
"""
function build_idf_dictionary(bow_matrix::Array{Int64, 2}, vocabulary::Dict{String, Int64},
    num_sentences::Int64)::Dict{String, Float64}

    # initialize -
    IDF_values_dictionary = Dict{String, Float64}();

    # compute smoothed IDF for each term -
    for (word, index) in vocabulary
        doc_frequency = sum(bow_matrix[:, index] .> 0);
        IDF_values_dictionary[word] = log((num_sentences + 1) / (doc_frequency + 1));
    end

    return IDF_values_dictionary;
end

"""
    build_tfidf_matrix(TF_matrix::Array{Float64, 2}, IDF_values_dictionary::Dict{String, Float64},
        inverse_vocabulary::Dict{Int64, String}) -> Array{Float64, 2}

Compute the TF-IDF matrix by multiplying TF values with corresponding IDF values.

### Arguments
- `TF_matrix::Array{Float64, 2}`: TF matrix of shape `(num_sentences, vocab_size)`.
- `IDF_values_dictionary::Dict{String, Float64}`: dictionary mapping words to IDF scores.
- `inverse_vocabulary::Dict{Int64, String}`: dictionary mapping indices to words.

### Returns
- `Array{Float64, 2}`: TF-IDF matrix of shape `(num_sentences, vocab_size)`.
"""
function build_tfidf_matrix(TF_matrix::Array{Float64, 2}, IDF_values_dictionary::Dict{String, Float64},
    inverse_vocabulary::Dict{Int64, String})::Array{Float64, 2}

    # initialize -
    num_sentences = size(TF_matrix, 1);
    vocab_size = size(TF_matrix, 2);
    TFIDF_matrix = zeros(Float64, num_sentences, vocab_size);

    # populate the TF-IDF matrix -
    for i in 1:num_sentences
        for j in 1:vocab_size
            word = inverse_vocabulary[j];
            idf_value = IDF_values_dictionary[word];
            TFIDF_matrix[i, j] = TF_matrix[i, j] * idf_value;
        end
    end

    return TFIDF_matrix;
end
