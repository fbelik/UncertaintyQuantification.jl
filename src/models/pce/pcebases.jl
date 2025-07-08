abstract type AbstractOrthogonalBasis end

struct PolynomialChaosBasis
    bases::Vector{<:AbstractOrthogonalBasis}
    p::Int
    d::Int
    α::Vector{Vector{Int64}}
end

function PolynomialChaosBasis(bases::Vector{<:AbstractOrthogonalBasis}, p::Int, α::Vector{Vector{Int}}=total_degree_set(p, length(bases)))
    @assert all(sum.(α) .<= p) "Multi-indices must have total degree at most p=$p"
    @assert all(α[1] .== 0) "First multi-index must be the 0-vector"
    d = length(bases)
    return PolynomialChaosBasis(bases, p, d, α)
end

function evaluate(Ψ::PolynomialChaosBasis, x::AbstractVector{Float64})
    res = ones(length(Ψ.α))
    for (i,α) in enumerate(Ψ.α)
        for (j,order) in enumerate(α)
            res[i] *= evaluate(Ψ.bases[j], x[j], order)
        end
    end
    return res
end

struct LegendreBasis <: AbstractOrthogonalBasis
    normalize::Bool
end

LegendreBasis() = LegendreBasis(true)

struct HermiteBasis <: AbstractOrthogonalBasis
    normalize::Bool
end

HermiteBasis() = HermiteBasis(true)

function evaluate(Ψ::AbstractOrthogonalBasis, x::AbstractVector{<:Real}, d::Int)
    return map(xᵢ -> evaluate(Ψ, xᵢ, d), x)
end

function evaluate(Ψ::LegendreBasis, x::Real, d::Int)
    val = P(x, d)
    return Ψ.normalize ? val * sqrt(2d + 1) : val
end

function evaluate(Ψ::HermiteBasis, x::Real, d::Int)
    val = He(x, d)
    return Ψ.normalize ? val / sqrt(factorial(d > 20 ? big(d) : d)) : val
end

function P(x, n::Integer)
    P⁻, P = zero(x), one(x)

    for i in 1:n
        P, P⁻ = ((2i - 1) * x * P - (i - 1) * P⁻) / i, P
    end
    return P
end

function He(x::Real, n::Integer)
    He⁻, He = zero(x), one(x)
    for i in 1:n
        He⁻, He = He, x * He - (i - 1) * He⁻
    end
    return He
end

function total_degree_set(p::Int, d::Int)
    No = binomial(p+d, d)

    idx = [zeros(Int, d) for _ in 1:No]
    idx[2][1] = 1
    cursum = 1

    for ct in 3:No
        idx[ct] .= idx[ct-1]
        cursum += 1
        idx[ct][1] += 1
        while cursum > p
            # Update multi-index
            for i in 1:d
                if idx[ct][i] > 0
                    cursum -= idx[ct][i]
                    idx[ct][i] = 0
                    if i < d
                        cursum += 1
                        idx[ct][i+1] += 1
                    end
                    break
                end
            end
        end
    end

    return idx
end

function hyperbolic_cross_set(p::Int, d::Int)
    idx = [zeros(Int, d) for _ in 1:2]
    idx[2][1] = 1
    curprod = 2

    while true
        if idx[end][end] == p
            break
        end
        push!(idx, copy(idx[end]))
        curidx = idx[end]
        curprod *= (curidx[1]+2) / (curidx[1]+1)
        curidx[1] += 1
        while curprod > (p+1)
            # Update multi-index
            for i in 1:d
                if curidx[i] > 0
                    curprod /= (curidx[i]+1)
                    curidx[i] = 0
                    if i < d
                        curprod *= (curidx[i+1]+2) / (curidx[i+1]+1)
                        curidx[i+1] += 1
                    end
                    break
                end
            end
        end
    end

    return idx
end

function q_norm_set(p::Int, d::Int, q::Real)
    if q == 1
        return total_degree_set(p, d)
    end

    idx = [zeros(Int, d) for _ in 1:2]
    idx[2][1] = 1
    curnormpowq = 1

    while true
        if idx[end][end] == p
            break
        end
        push!(idx, copy(idx[end]))
        curidx = idx[end]
        curnormpowq += (curidx[1]+1)^q - (curidx[1])^q
        curidx[1] += 1
        while (curnormpowq - 1e-4) > (p^q + eps())
            # Update multi-index
            for i in 1:d
                if curidx[i] > 0
                    curnormpowq -= (curidx[i])^q
                    curidx[i] = 0
                    if i < d
                        curnormpowq += (curidx[i+1]+1)^q - (curidx[i+1])^q
                        curidx[i+1] += 1
                    end
                    break
                end
            end
        end
    end

    return idx
end

function map_to_base(_::LegendreBasis, x::AbstractVector)
    return quantile.(Uniform(-1, 1), cdf.(Normal(), x))
end

function map_to_base(_::HermiteBasis, x::AbstractVector)
    return x
end

function map_to_bases(Ψ::PolynomialChaosBasis, x::AbstractMatrix)
    return mapreduce((b, xᵢ) -> map_to_base(b, xᵢ), hcat, Ψ.bases, eachcol(x))
end

function map_from_base(_::LegendreBasis, x::AbstractVector)
    return quantile.(Normal(), cdf.(Uniform(-1, 1), x))
end

function map_from_base(_::HermiteBasis, x::AbstractVector)
    return x
end

function map_from_bases(Ψ::PolynomialChaosBasis, x::AbstractMatrix)
    return hcat(map((b, xᵢ) -> map_from_base(b, xᵢ), Ψ.bases, eachcol(x))...)
end

function quadrature_nodes(n::Int, _::LegendreBasis)
    x, _ = gausslegendre(n)
    return x
end

function quadrature_weights(n::Int, _::LegendreBasis)
    _, w = gausslegendre(n)
    return w ./ 2
end

function quadrature_nodes(n::Int, _::HermiteBasis)
    x, _ = gausshermite(n)
    return sqrt(2) * x
end

function quadrature_weights(n::Int, _::HermiteBasis)
    _, w = gausshermite(n)
    return w / sqrt(π)
end

function sample(n::Int, _::LegendreBasis)
    return rand(Uniform(-1, 1), n)
end

function sample(n::Int, _::HermiteBasis)
    return rand(Normal(), n)
end
