"""
    generate_thrombin_dataset(tf_values::Vector{Float64};
        tspan::Tuple{Float64,Float64} = (0.0, 1200.0), saveat::Float64 = 5.0) -> Tuple

Generate a dataset of thrombin generation curves by simulating the Hockin-Mann 2002 coagulation
model at each tissue factor (TF) concentration in `tf_values`.

### Arguments
- `tf_values::Vector{Float64}`: vector of TF concentrations in molar (M).
- `tspan::Tuple{Float64,Float64}`: simulation time span in seconds (default: 0 to 1200).
- `saveat::Float64`: time step for saving solution points in seconds (default: 5.0).

### Returns
- `Tuple{Vector{Float64}, Matrix{Float64}, Vector{Float64}}`: tuple of (time_vector, thrombin_matrix, tf_values)
   where `thrombin_matrix` has rows as time points and columns as different TF concentrations.
   Thrombin values are in nanomolar (nM).
"""
function generate_thrombin_dataset(tf_values::Vector{Float64};
    tspan::Tuple{Float64,Float64} = (0.0, 1200.0), saveat::Float64 = 5.0)::Tuple{Vector{Float64}, Matrix{Float64}, Vector{Float64}}

    # initialize -
    n_curves = length(tf_values);
    sol_first = HockinMannModel.simulate(HockinMann2002; TF_concentration=tf_values[1], tspan=tspan, saveat=saveat);
    time_vector = sol_first.t;
    n_timepoints = length(time_vector);
    thrombin_matrix = zeros(Float64, n_timepoints, n_curves);

    # simulate each TF concentration -
    for (j, tf) in enumerate(tf_values)
        sol = HockinMannModel.simulate(HockinMann2002; TF_concentration=tf, tspan=tspan, saveat=saveat);
        thrombin_matrix[:, j] = HockinMannModel.total_thrombin(HockinMann2002, sol) .* 1e9; # convert M to nM
    end

    return (time_vector, thrombin_matrix, tf_values)
end

"""
    normalize_minmax(data::Matrix{Float64}) -> Tuple{Matrix{Float64}, Float64, Float64}

Apply min-max normalization to scale all values in `data` to the interval [0, 1].

### Arguments
- `data::Matrix{Float64}`: matrix of values to normalize.

### Returns
- `Tuple{Matrix{Float64}, Float64, Float64}`: tuple of (normalized_data, data_min, data_max).
"""
function normalize_minmax(data::Matrix{Float64})::Tuple{Matrix{Float64}, Float64, Float64}
    data_min = minimum(data);
    data_max = maximum(data);
    normalized = (data .- data_min) ./ (data_max - data_min);
    return (normalized, data_min, data_max)
end

"""
    denormalize_minmax(data::Matrix{Float64}, data_min::Float64, data_max::Float64) -> Matrix{Float64}

Reverse min-max normalization to recover original-scale values.

### Arguments
- `data::Matrix{Float64}`: normalized matrix with values in [0, 1].
- `data_min::Float64`: minimum value from the original data.
- `data_max::Float64`: maximum value from the original data.

### Returns
- `Matrix{Float64}`: data in original scale.
"""
function denormalize_minmax(data::Matrix{Float64}, data_min::Float64, data_max::Float64)::Matrix{Float64}
    return data .* (data_max - data_min) .+ data_min
end

"""
    denormalize_minmax(data::Vector{Float64}, data_min::Float64, data_max::Float64) -> Vector{Float64}

Reverse min-max normalization for a vector.

### Arguments
- `data::Vector{Float64}`: normalized vector with values in [0, 1].
- `data_min::Float64`: minimum value from the original data.
- `data_max::Float64`: maximum value from the original data.

### Returns
- `Vector{Float64}`: vector in original scale.
"""
function denormalize_minmax(data::Vector{Float64}, data_min::Float64, data_max::Float64)::Vector{Float64}
    return data .* (data_max - data_min) .+ data_min
end
