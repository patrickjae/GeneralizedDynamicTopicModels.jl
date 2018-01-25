import Base.exp
"""
computes log(x+y) given a = log(x) and b = log(y)
"""
log_add(a::Union{Float64, Array{Float64}},b::Union{Float64, Array{Float64}}) = max(a,b) + log(1 + exp(-abs(a-b)))

"""
Inplace version of log_add, stores log(x+y) into a and expects a = log(x), b = log(y) as inputs.
"""
function log_add!(a::Vector, i::Int64, b::Float64)
	a[i] = max(a[i],b) + safe_log(1 + exp(-abs(a[i]-b)))
end

function log_add2!(a::Vector, i::Int64, b::Float64)
	if a[i] < b
		a[i] = b + log(1+exp(a[i]-b))
	else
		a[i] = a[i] + log(1+exp(b-a[i]))
	end
end

function log_add!(a::Matrix, i::Int64, j::Int64, b::Float64)
	a[i,j] = max(a[i,j],b) + safe_log(1 + exp(-abs(a[i,j]-b)))
end

function log_add2!(a::Matrix, i::Int64, j::Int64, b::Float64)
	if a[i,j] < b
		a[i,j] = b + log(1+exp(a[i,j]-b))
	else
		a[i,j] = a[i,j] + log(1+exp(b-a[i,j]))
	end
end
"""
computes the sum log(a_1 + ... + a_n) given the array of
[log(a_1), ..., log(a_n)]
"""
function log_sum(a::Array{Float64, 1})
	r = a[1]
	for i in 2:length(a)
		r = log_add(r, a[i])
	end
	r
end

"""
computes log(x-y) given a = log(x) and b = log(y)
"""
function log_subtract(a::Float64,b::Float64)
	if a < b
		return -1000.0
	end
	a + safe_log(1 - exp(b-a))
end

function log_subtract(a::Array{Float64, 1},b::Array{Float64, 1})
	r = zeros(length(a))
	for i in 1:length(a)
		r[i] = log_subtract(a[i], b[i])
	end
	r
end

function log_subtract!(a::Matrix, col::Int64, b::Vector)
	for i in 1:length(b)
		a[i, col] = log_subtract(a[i,col], b)
	end
end

function safe_log(a)
	if a < 0
		return -1000.0
	end
	log(a + 1e-100)
end

"""
Computes m'*diagm(d)*m efficiently.
"""
function matrix_diagonal_product(m::Matrix, d::Vector)
	B = broadcast(.*, m, sqrt(d))
	B'*B
end

function chol_inv(m::Matrix{Float64})
	try
		L_inv = inv(chol(Hermitian(m))')
		return L_inv'*L_inv, prod(diag(L_inv))^2
	catch
		return inv(m), 1./det(m)
	end
end


function inverse_and_determinant(m::Matrix)
	(l,u,p) = lu(m)
	l = LowerTriangular(l)
	u = UpperTriangular(u)
	n = size(m,1)
	e_vec = zeros(n)
	inv_m = zeros(n,n)
	S = 0
	for i in 1:n
		e_vec[i] = 1.
		if i > 1 e_vec[i-1] = 0. end
		if p[i] != i S += 1 end
		y = l \ e_vec
		inv_m[p[i],:] = u\y
	end
	(inv_m, 1./(det(l)*det(u)*(-1)^S))
end

"""
Computes the inverse inv(A+B) in terms of inv(A) and B.
Based on the algorithm described in:
Miller, K. S. (1981). On the Inverse of the Sum of Matrices. Mathematics Magazine, 54(2), 67–72.

The above is simplified to the case where H is diagonal and hence each decomposition
is a multiple of the unit vector in the according dimension.
"""
function matrix_sum_inverse(a_inverse::Matrix, b::Matrix, det_a::Float64)
	#assume that b is a diagonal matrix (H in the paper)
	num_ranks = size(b,1)
	#C_1 is G
	C_k_inv = a_inverse
	result_det = det_a
	for r in 1:num_ranks
		#extract rank 1 decomposition
		# tmp =
		v_k = 1./(1+C_k_inv[r,r]*b[r,r])
		result_det /= v_k
		#prepare for next step
		C_k_inv -= v_k*C_k_inv.*b[:,r]'*C_k_inv
	end
	#return last C_k_inv = inv(G+H) and determinant of the inverse
	(C_k_inv, 1./result_det)
end

function matrix_sum_inverse2(a_inverse::Array{Float64,2}, decomposition::Array{Array{Float64,2},1})
	num_ranks = length(decomposition)
	C_k_inv = a_inverse
	result_det = prod(diag(a_inverse))
	for r in 1:num_ranks
		tmp = C_k_inv*decomposition[r]
		v_k = 1./(1+trace(tmp))
		result_det *= v_k
		C_k_inv -= v_k*tmp*C_k_inv
	end
	(C_k_inv, result_det)
end


function get_topic_distributions(m::GDTM)
	topic_probabilities = Vector{Matrix{Float64}}(m.K)
	for k in 1:m.K
		means = m.KnmKmmInv*m.μ[k]
		topic_probabilities[k] = zeros(m.T, m.V)
		Threads.@threads for w in 1:m.V
			topic_probabilities[k][:,w] = exp(means[:,w] - m.zeta[k,:])
		end
	end
	topic_probabilities
end

function get_topic_distributions(path::String, save_images::Bool=false)
	no_zeta = !isfile(joinpath(path, "zeta.mat"))
	if !no_zeta
		zeta_file = open(joinpath(path, "zeta.mat"), "r")
		zeta = read_array(zeta_file)
		close(zeta_file)
	end
	tp_raw = read_topic_distribution(joinpath(path, "topics.dat"))

	times = map(x->parse(Float64, x), readlines(joinpath(path, "times.dat")))
	inducing_times = map(x->parse(Float64, x), readlines(joinpath(path, "inducing_times.dat")))
	krn_file = open(joinpath(path, "kernel.krn"), "r")
	krn = deserialize(krn_file)
	close(krn_file)
	if length(times) < 50
		test_times = collect(linspace(times[1],times[end], 250))
		times = test_times
		no_zeta=true
	end
	Knm = Kernels.computeCrossCovarianceMatrix(krn, times, inducing_times)
	Kmm = Kernels.computeCovarianceMatrix(krn, inducing_times)
	KnmKmmInv = Knm*pinv(Kmm)

	for k in 1:length(tp_raw)
		tp_raw[k] = KnmKmmInv*tp_raw[k]
	end

	for k in 1:length(tp_raw)
		(T,V) = size(tp_raw[k])
		if no_zeta
			tp_raw[k] = exp(tp_raw[k])
			tp_raw[k] ./= vec(sum(tp_raw[k],2))
		else
			for w in 1:V
				tp_raw[k][:,w] = exp(tp_raw[k][:,w] - zeta[k,:])
			end
		end
	end
	tp_raw
end

function write_topic_words(m::GDTM, corpus::CorpusUtils.Corpus, target_file::String)
	topic_probabilities = get_topic_distributions(m)
	#extract top words for topics
	f = open(target_file, "w")
	for k in 1:m.K
		@printf(f, "topic %d\n", k)
		for t in 1:m.T
			cur_topic = vec(topic_probabilities[k][t,:])
			sorted_idx = sortperm(cur_topic,rev=true)
			@printf(f, "%s:\t", Dates.format(m.timestamps[t], "yyyy-mm-dd"))
			[@printf(f, "%s ", corpus.lexicon.words[word].value) for word in sorted_idx[1:10]]
			@printf(f, "\n")
		end
		@printf(f, "\n")
	end
	close(f)
end

function write_stats(m::GDTM; base_dir_suffix::String="", topics_only::Bool=false)
	base_dir = "experiments/$base_dir_suffix/$(split(string(typeof(m.krn)), ".")[2])$(m.krn.params)/alpha$(m.alpha)/datavar$(m.s_x)_prior_mean$(m.m_0)_prior_var$(m.s_0)/$(m.K)_topics"
	mkpath(base_dir)

	write_topic_words(m, m.corpus, joinpath(base_dir, "topics.txt"))
	if topics_only
		return 
	end
	elbo_file = open(joinpath(base_dir,"elbo"), "w")
	lr_file = open(joinpath(base_dir,"learning_rates"), "w")
	for i in 1:length(m.likelihood_counts)
		@printf(elbo_file, "%f\t%f\n", m.likelihood_counts[i], m.likelihoods[i])
		@printf(lr_file, "%f\t%f\n", m.likelihood_counts[i], m.learning_rates[i])
	end
	close(elbo_file)
	close(lr_file)
	test_file = open(joinpath(base_dir,"test"), "w")
	for i in 1:length(m.test_counts)
		@printf(test_file, "%f\t%f\n", m.test_counts[i], m.test_perplexities[i])
	end
	close(test_file)
	times_file = open(joinpath(base_dir, "times.dat"), "w")
	times_to_store = map(Dates.datetime2julian,m.timestamps)
	for t in 1:m.T
		@printf(times_file, "%f\n", times_to_store[t])
	end
	close(times_file)
	inducing_times_file = open(joinpath(base_dir, "inducing_times.dat"), "w")
	for i in m.inducing_points
		@printf(inducing_times_file, "%f\n", i)
	end
	close(inducing_times_file)
	zeta_file = open(joinpath(base_dir, "zeta.mat"), "w")
	write_array(m.zeta, zeta_file)
	close(zeta_file)
	kernel_file = open(joinpath(base_dir, "kernel.krn"), "w")
	serialize(kernel_file, m.krn)
	close(kernel_file)
	write_topic_distribution(m, base_dir)
	write_document_distribution(m, base_dir)
	base_dir
end


function write_topic_distribution(m::GDTM, base_dir::String)
	#write the topics
	f = open(joinpath(base_dir, "topics.dat"), "w")
	#write num topics
	write(f, m.K)
	for k in 1:m.K
		write_array(m.μ[k], f)
	end
	close(f)
end

function read_topic_distribution(td_file::String)
	f = open(td_file, "r")
	K = read(f, Int64)
	tp = Vector{Array}(K)
	for k in 1:K
		tp[k] = read_array(f)
	end
	close(f)
	tp
end

function write_document_distribution(m::GDTM, base_dir::String)
	f = open(joinpath(base_dir, "documents.dat"), "w")
	# get distributions over topics for all documents
	(_, _, _, _, _, document_proportions) = e_step(m, collect(1:m.D))
	# write the array
	write_array(document_proportions, f)
	close(f)
end

function read_document_distribution(dd_file::String)
	f = open(dd_file, "r")
	dd = read_array(f)
	close(f)
	dd
end

function write_array(a, f)
	s = size(a)
	write(f, length(s))
	write(f, collect(s))
	write(f, a)
end

function read_array(f)
	num_dims = read(f, Int64)
	dims = tuple(read(f, Int64, num_dims)...)
	read(f, Float64, dims)
end


function tab_split(file::String)
	f = open(file,"r")
	c = Float64[]
	v = Float64[]
	for l in readlines(f)
		(cur_c, cur_v) = split(chomp(l), '\t')
		push!(c, parse(Float64, cur_c))
		push!(v, parse(Float64, cur_v))
	end
	close(f)
	(c,v)
end
