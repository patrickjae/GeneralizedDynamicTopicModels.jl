module Kernels
# super type
abstract type Kernel end

# diferent kernels to use
type RBFKernel <: Kernel
	params::Array{Real,1}
	RBFKernel(s2, l) = new([s2,l])
	RBFKernel(params) = new(params)
end

type BrownianKernel <: Kernel
	params::Array{Float64,1}
	BrownianKernel(s2::Float64) = new([s2])
end

type OrnsteinUhlenbeckKernel <: Kernel
	params::Array{Float64,1}
	OrnsteinUhlenbeckKernel(s2::Float64, l::Float64) = new([s2, l])
end

type WhiteNoiseKernel <: Kernel
	params::Array{Float64,1}
	WhiteNoiseKernel(s2) = new([s2])
end

type PeriodicRBFKernel <: Kernel
	params::Vector
	PeriodicRBFKernel(s2::Float64, l::Float64, p::Float64) = new([s2,l,p])
end

type CauchyKernel <: Kernel
	params::Vector
	CauchyKernel(s2::Float64, scale::Float64) = new([s2, scale])
end

type RationalQuadraticKernel <: Kernel
	params::Vector
	RationalQuadraticKernel(s2::Float64, l::Float64, alpha::Float64) = new([s2, l, alpha])
end

type ConstantKernel <: Kernel
	params::Vector
	ConstantKernel(s2::Float64) = new([s2])
end

export Kernel,
	RBFKernel,
	BrownianKernel,
	OrnsteinUhlenbeckKernel,
	WhiteNoiseKernel,
	PeriodicRBFKernel,
	CauchyKernel,
	computeCovarianceMatrix,
	computeCrossCovarianceMatrix



# covariance function for various kernels
# rbf
computeCovariance(k::RBFKernel, x::Array{Float64, 1}, x_prime::Array{Float64, 1}) = k.params[1] * exp(-.5*norm(x-x_prime)^2/(k.params[2]^2))

# brownian motion
computeCovariance(k::BrownianKernel, x::Array{Float64, 1}, x_prime::Array{Float64, 1}) = float((k.params[1] * min(x, x_prime))[1])

# ornstein-uhlenbeck
computeCovariance(k::OrnsteinUhlenbeckKernel, x::Array{Float64, 1}, x_prime::Array{Float64, 1}) = k.params[1] * exp(-norm(x-x_prime)/k.params[2])

# white noise
computeCovariance(k::WhiteNoiseKernel, x::Array{Float64, 1}, x_prime::Array{Float64, 1}) = x==x_prime ? k.params[1] : 0

#periodic
computeCovariance(k::PeriodicRBFKernel, x::Vector, x_prime::Vector) = k.params[1] * exp(-2*sin(norm(x-x_prime)/k.params[3])^2/(k.params[2]^2) )

#cauchy
computeCovariance(k::CauchyKernel, x::Vector, x_prime::Vector) = k.params[1]/(1 + (norm(x-x_prime)/k.params[2])^2)

#rational quadratic
computeCovariance(k::RationalQuadraticKernel, x::Vector, x_prime::Vector) = k.params[1]*(1 + (norm(x-x_prime)^2 / (2*k.params[3]*k.params[2]^2)))^(-k.params[3])

#constant
computeCovariance(k::ConstantKernel, x::Vector, x_prime::Vector) = k.params[1]

# to_string(k::Kernel) = 


function setParameters{T<:Real}(k::Kernel, newParams::Array{T, 1})
	k.params[:] = newParams[:]
end

#compute K_xx
function computeCovarianceMatrix{T}(k::Kernel, points::Array{T})
	n = size(points)[1]
	cov = zeros(n,n)
	for i in 1:n, j in i:n
		cov[i,j] = computeCovariance(k, vec(points[i,:]), vec(points[j,:]))
		cov[j,i] = cov[i,j]
	end
	cov
end



#compute K_xy
function computeCrossCovarianceMatrix{T}(k::Kernel, x::Array{T}, x_prime::Array{T})
	n = size(x)[1]
	m = size(x_prime)[1]
	cov = zeros(n,m)
	for i in 1:n, j in 1:m
		cov[i,j] = computeCovariance(k, vec(x[i,:]), vec(x_prime[j,:]))
	end
	cov
end


end
