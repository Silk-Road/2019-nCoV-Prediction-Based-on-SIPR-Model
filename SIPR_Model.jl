# encoding = utf-8
# Author: Liu Jing
# Email: zhulincaowu@gmail.com

using Distances, Distributions, DataFrames

#----------------------------------------------------
#                       SIPR模型
#----------------------------------------------------
start_time = 0
end_time = 180
Δt = 1
n = Int(ceil(end_time - start_time) / Δt)
b = 0.6489
ν₁ = 0.0836
ν₂ = 0.0794
β = 0.2586

function de_s(β, s, i, p)
    return -β * (i + p) * s
end

function de_i(b, β, i, p, s, ν₁)
    return b * β * (i + p) * s - ν₁ * i
end

function de_p(b, β, i, p, s, ν₂)
    return (1 - b) * β * (i + p) * s - ν₂ * p
end

function de_r(ν₁, ν₂, i, p)
    return ν₁ * i + ν₂ * p
end

function sipr(β, b, ν₁, ν₂, n, i0, Δt=1)
    """
    SIPR model
    """
    S = repeat([0.0], n)
    I = repeat([0.0], n)
    P = repeat([0.0], n)
    R = repeat([0.0], n)
    total = repeat([0.0], n)

    S[1] = 1400000000.0
    I[1] = i0
    P[1] = 0.0
    R[1] = 0.0

    N = S[1] + I[1] + P[1] + R[1]
    β = β/N
    total[1] = N

    for i in 2:n
        S[i] = S[i-1] + de_s(β, S[i-1], I[i-1], P[i-1]) * Δt
        I[i] = I[i-1] + de_i(b, β, I[i-1], P[i-1], S[i-1], ν₁) * Δt
        P[i] = P[i-1] + de_p(b, β, I[i-1], P[i-1], S[i-1], ν₂) * Δt
        R[i] = R[i-1] + de_r(ν₁, ν₂, I[i-1], P[i-1]) * Δt
        total[i] = S[i] + I[i] + P[i] + R[i]
    end

    df = DataFrame(S = S, I = I, P = P, R = R, N = total)
    return df
end

df_sipr = sipr(β, b, ν₁, ν₂, n, 1)
df_sipr[60:70,:]

#----------------------------------------------------
#                 计算SIPR模型参数
#----------------------------------------------------

number_samples = 10000000
β_prior = .^(10,rand(Uniform(-1,0), number_samples))
b_prior = rand(Uniform(0,1), number_samples)
ν₁_prior = rand(Uniform(0,0.1), number_samples)
ν₂_prior = rand(Uniform(0,0.1), number_samples)
i0_prior = rand(Poisson(1), number_samples).+1

N = 1400000000 # 人口总数
time_range = 100 # 100天传染扩散, time_range == length(observed_cases)
threshold = 20000 # 用以存储后验的阀值

β_posterior = Float64[]
b_posterior = Float64[]
ν₁_posterior = Float64[]
ν₂_posterior = Float64[]
i0_posterior = Int64[]
distance_posterior = Float64[]

observed_cases = df_sipr[:I][1:100] # 基于之前论文参数模拟的数据作为观测数据


for i in 1:number_samples
    simulated_timeseries = sipr(β_prior[i], b_prior[i], ν₁_prior[i],
                               ν₂_prior[i], time_range, i0_prior[i])
    distance = evaluate(Euclidean(), observed_cases, simulated_timeseries[2])
    if distance <= threshold
        push!(β_posterior, β_prior[i])
        push!(b_posterior, b_prior[i])
        push!(ν₁_posterior, ν₁_prior[i])
        push!(ν₂_posterior, ν₂_prior[i])
        push!(i0_posterior, i0_prior[i])
        push!(distance_posterior, distance)
    end
end

posteriors = DataFrame(β = β_posterior,
                       ν₁ = ν₁_posterior,
                       ν₂ = ν₂_posterior,
                       b = b_posterior,
                       i0 = i0_posterior)


beta = mean(posteriors[:β])
nu_1 = mean(posteriors[:ν₁])
nu_2 = mean(posteriors[:ν₂])
b = mean(posteriors[:b])
i0 = mean(posteriors[:i0])

# SIPR R0计算详见论文：https://pdfs.semanticscholar.org/a239/12ed014f492bb54769ce0497c5f0b84499e7.pdf
R0 = (1 - b) * β / ν₂ + b * β / ν₁

simulated_timeseries = sipr(β, b, ν₁, ν₂, time_range, i0)
simulated_timeseries[60:70,:]
