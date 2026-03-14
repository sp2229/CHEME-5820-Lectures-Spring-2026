"""
    build_bow_matrix(sentences::Array{String,1}, vocabulary::Dict{String, Int64}) -> Array{Int64, 2}

Build a Bag of Words matrix from a corpus of sentences and a vocabulary.
Each row corresponds to a sentence and each column corresponds to a word in the vocabulary.
Sentences are augmented with `<bos>` and `<eos>` tokens, converted to lowercase, and split by whitespace.
Words not in the vocabulary are mapped to the `<unk>` token.

### Arguments
- `sentences::Array{String,1}`: array of sentence strings.
- `vocabulary::Dict{String, Int64}`: dictionary mapping words to vocabulary indices.

### Returns
- `Array{Int64, 2}`: matrix of shape `(num_sentences, vocab_size)` with word counts.
"""
function build_bow_matrix(sentences::Array{String,1}, vocabulary::Dict{String, Int64})::Array{Int64, 2}

    # initialize -
    num_sentences = length(sentences);
    vocab_size = length(vocabulary);
    bow_matrix = zeros(Int64, num_sentences, vocab_size);

    # populate the Bag of Words matrix -
    for (i, sentence) in enumerate(sentences)
        augmented_sentence = "<bos> " * sentence * " <eos>";
        words = split(lowercase(augmented_sentence)) .|> String;
        for word in words
            if haskey(vocabulary, word)
                index = vocabulary[word];
                bow_matrix[i, index] += 1;
            else
                unk_index = vocabulary["<unk>"];
                bow_matrix[i, unk_index] += 1;
            end
        end
    end

    return bow_matrix;
end

"""
    hashing_vectorizer(features::Array{String,1}; length::Int64 = 10) -> Array{Int64, 1}

Convert a list of features (words) into a fixed-length vector using the hashing trick.
Each feature is mapped to an index using Julia's built-in `hash` function, and the count at that index is incremented.

### Arguments
- `features::Array{String,1}`: array of feature strings (e.g., words from a sentence).
- `length::Int64`: desired length of the output vector (default: 10).

### Returns
- `Array{Int64, 1}`: vector of length `length` with hashed feature counts.
"""
function hashing_vectorizer(features::Array{String,1}; length::Int64 = 10)::Array{Int64,1}

    # initialize -
    new_hash_vector = zeros(Int, length);
    for i ∈ eachindex(features)
        feature = features[i];
        j = hash(feature) |> h -> mod1(h, length);
        new_hash_vector[j] += 1;
    end

    return new_hash_vector;
end
