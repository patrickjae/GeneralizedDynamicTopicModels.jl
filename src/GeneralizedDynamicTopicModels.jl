include("kernels.jl")
include("dynamicdata/DynamicData.jl")
include("corpusutils/CorpusUtils.jl")
module GeneralizedDynamicTopicModels
using GeneralizedDynamicTopicModels, Distributions, Plots, Kernels, DynamicData, CorpusUtils
typealias DocumentData Union{CorpusUtils.DocumentList, DynamicData.DynamicDataSet}
type GDTM
	m_0::Float64 #prior mean
	s_0::Float64 #prior variance
	s_x::Float64 #measurement variance
	invS #inverse variance
	eta_1::Vector{Matrix{Float64}} #natural parameters to the topic normals
	eta_2::Matrix{Matrix{Float64}}
	alpha::Float64 #alpha prior for dirichlets
	corpus::CorpusUtils.Corpus #the data
	training_doc_ids::Vector{Int64} #ids of the training documents
	validation_doc_ids::Vector{Int64} # ids of validation documents
	test_doc_ids::Vector{Int64} # ids of test documents
	T::Int64 #number of time points
	D::Int64 #number of (training) documents
	K::Int64 #number of topics
	V::Int64 #number of words
	batch_size::Int64 #size of minibatch
	
	lambda::Matrix{Float64} #variational parameter to the dirichlets
	phi::Vector{Matrix{Float64}} #variational parameter to the multinomials (selecting the source distribution)
	zeta::Matrix{Float64} #variational parameter for bounding the intractable expectation caused by softmax fct
	suff_stats_tk::Vector{Vector{Float64}} #suffstats
	suff_stats_tkx::Vector{Matrix{Float64}} #suffstats
	means::Vector{Matrix{Float64}} #variational means
	s::Matrix{Matrix{Float64}} #variational variance
	timestamps
	times::Vector{Float64} #time points
	visualize::Bool # switch on or off any visualization
	inducing_points::Vector{Float64} #inducing point locations for sparse GP
	Kmm::Hermitian{Float64, Matrix{Float64}} #inducing point covariance for sparse GP
	KmmInv::Hermitian{Float64, Matrix{Float64}} #inverse
	Knn::Hermitian{Float64, Matrix{Float64}} #full rank covariance for GP models
	KnnInv::Hermitian{Float64, Matrix{Float64}} #inverse
	Knm::Matrix{Float64} #cross covariance training points - inducing points for sparse GP
	KnmKmmInv::Matrix{Float64} #cross covariance x inverse inducing point covariance, to save computation
	K_tilde::Hermitian{Float64, Matrix{Float64}} #low rank approximation of full rank covariance (sparse GP)
	K_tilde_diag::Vector{Float64} #diagonal of K_tilde
	S_diags::Vector{Matrix{Float64}} #variational covariance diagonals
	Λ_diags::Vector{Matrix{Float64}} #helper for storing Λ diagonals
	μ::Vector{Matrix{Float64}} #inducing point values in sparse GP
	krn::Kernels.Kernel #kernel object for GP models
	likelihood_counts::Vector{Float64} #list of #docs seen for likelihoods (ELBO estimates)
	test_counts::Vector{Float64} #same but for test set likelihoods
	likelihoods::Vector{Float64} #measured ELBO estimates
	test_perplexities::Vector{Float64} #measured test set likelihoods
	learning_rates::Vector{Float64} #computed learning rates
	word_observation_times::Vector{Vector{Int64}}
	jitter::Diagonal{Float64}
	use_seeding::Bool
	function GDTM(corpus::CorpusUtils.Corpus, num_topics::Int64, batch_size::Int64, 
					m_0::Float64, s_0::Float64, s_x::Float64, alpha::Float64, visualize::Bool; 
					training_doc_ids::Vector{Int64}=Int64[], validation_doc_ids::Vector{Int64}=Int64[], 
					test_doc_ids::Vector{Int64}=Int64[], seeded::Bool = false)
		n = DynamicData.length(corpus.documents)
		m = new()
		m.batch_size = batch_size
		m.timestamps = DynamicData.get_unique_timestamps(corpus.documents)
		times = map(Dates.datetime2julian,m.timestamps)
		times -= minimum(times)-1
		m.times = times
		m.T = length(times)
		m.visualize = visualize
		m.use_seeding = seeded

		#just for testing
		suggested_variance = 1./(times[end] - times[1])
		#the data
		m.corpus = corpus
		#if nothing set, assume we need only test data
		if length(training_doc_ids) == 0
			r = randperm(n)
			test_doc_num = min(round(Int64, n*.1), 10000)
			info("creating test set, using $test_doc_num for testing, $(length(r)-test_doc_num) for training")
			m.test_doc_ids = r[1:test_doc_num]
			m.training_doc_ids = r[test_doc_num+1:end]
		else
			m.training_doc_ids = training_doc_ids
		end
		if length(test_doc_ids) != 0
			m.test_doc_ids = test_doc_ids
		end
		m.D = length(m.training_doc_ids)
		m.K = num_topics
		m.V = length(corpus.lexicon)
		m.word_observation_times = Vector{Vector{Int64}}(m.V)
		[m.word_observation_times[w] = collect(CorpusUtils.used_in_times(m.corpus, m.corpus.lexicon.words[w])) for w in 1:m.V]

		#model params
		m.m_0 = m_0
		m.s_0 = s_0

		#measurement noise
		m.s_x = s_x
		#document dirichlet param
		m.alpha = alpha
		reset(m)
		Plots.plotly()
		println("done constructor")
		m
	end
end

include("utils.jl")
include("gpdtm_svi_inference.jl")
include("visualize.jl")

function reset(m::GDTM)
	#stats
	m.likelihood_counts = Float64[]
	m.test_counts = Float64[]
	m.likelihoods = Float64[]
	m.test_perplexities = Float64[]
	m.learning_rates = Float64[]

	m.timestamps = DynamicData.get_unique_timestamps(m.corpus.documents)
	times = map(Dates.datetime2julian,m.timestamps)
	times -= minimum(times)-1
	m.times = times
	m.T = length(times)
	# reset zeta parameter
	m.zeta = zeros(m.K, m.T)
	# garbage collection
	gc()
end


"""
The update step for local, i.e. document specific variational parameters. Computes the likelihood of the processed documents on the fly.
"""
function e_step(m::GDTM, data_idx::Vector{Int64})
	e_step_ll = 0
	words_seen = Int64[]
	[words_seen = union(words_seen, map(w->w.id,collect(keys(m.corpus.documents[doc_id].value)))) for doc_id in data_idx]
	
	#reset sufficient statistics for the minibatch
	ɸ = Array(Array{Float64, 1},m.K)
	Ξ = Array(Array{Float64, 2},m.K)
	for k in 1:m.K
		ɸ[k] = zeros(m.T)
		Ξ[k] = zeros(m.T, m.V)
	end
	timestamps_seen = IntSet()
	doc_topic_proportions = zeros(length(data_idx), m.K)
	for (i, doc_idx) in enumerate(data_idx)
		(doc_ll, λ) = doc_e_step(m, doc_idx, ɸ, Ξ, timestamps_seen)
		e_step_ll += doc_ll
		doc_topic_proportions[i,:] = λ
	end
	(e_step_ll, ɸ, Ξ, words_seen, collect(timestamps_seen), doc_topic_proportions)
end


"""
Performs the e-step on individual documents.
"""
function doc_e_step(m::GDTM, doc_idx::Int64, ɸ::Array{Vector{Float64},1}, Ξ::Array{Matrix{Float64},1}, timestamps_seen::IntSet)
	doc = m.corpus.documents[doc_idx]
	#extract wird ids and frequencies from the document
	(doc_words, freqs) = CorpusUtils.get_words_and_counts(doc.value)
	# extract timestamp of the documents
	t_doc = DynamicData.get_timestamp_index(m.corpus.documents, doc.timestamp)
	push!(timestamps_seen, t_doc)
	# do the actual inference
	(φ,λ,vi_ll) = document_inference(m, t_doc, doc_words, freqs)
	#collect sufficient statistics
	for k in 1:m.K
		for (i,w) in enumerate(doc_words)
			ɸ[k][t_doc] += freqs[i]*φ[i,k]
			Ξ[k][t_doc, w] += freqs[i]*φ[i,k]
		end
	end
	(vi_ll, λ)
end

"""
Perform variational inference on document parameters, i.e. proportions of topics in the document and single word assignment probabilities.
"""
function document_inference(m, t_doc::Int64, doc_words::Vector{Int64}, freqs::Vector{Int64})
	last_vi_ll = -1e100
	vi_ll = 0
	converged = 1
	#get document data
	N_d = length(doc_words)
	#init phi variable for doc
	φ = ones(length(doc_words), m.K) ./ m.K
	#init lambda for doc
	λ = ones(m.K) .* m.alpha .+ N_d./m.K
	dig_λ = digamma(λ)
	dig_λ_sum = digamma(sum(λ))

	iter = 0
	means = zeros(m.K, N_d)
	# compute means for only those words that are observed in the document from inducing point means
	for k in 1:m.K
		means[k,:] = m.KnmKmmInv[t_doc,:]'*m.μ[k][:, doc_words]
	end
	# do inference
	while converged > 1e-3
		vi_ll = 0
		for i in 1:N_d
			log_phi_sum = 0
			for k in 1:m.K
				φ[i,k] = (dig_λ[k] - dig_λ_sum) + (means[k, i] - m.zeta[k, t_doc])

				log_phi_sum = k==1 ? φ[i,k] : log_add(φ[i,k], log_phi_sum)
			end
			#normalize and exp phi
			φ[i,:] = exp(φ[i,:] .- log_phi_sum) + 1e-100
		end
		#update lambda
		λ = m.alpha .+ vec(sum(freqs.*φ,1))
		dig_λ = digamma(λ)
		dig_λ_sum = digamma(sum(λ))

		#compute the document likelihood given the current parameters
		vi_ll = compute_document_likelihood(m, t_doc, doc_words, freqs, φ, λ, means)
		#check convergence
		converged = (last_vi_ll - vi_ll)/last_vi_ll
		last_vi_ll = vi_ll
		iter += 1
	end
	(φ, λ, vi_ll)
end


"""
Computes the document likelihood given some paramter setting.
"""
function compute_document_likelihood(m::GDTM, t_doc::Int64, doc_words::Vector{Int64}, freqs::Vector{Int64}, φ::Matrix{Float64}, λ::Vector{Float64}, means::Matrix{Float64})
	likelihood = 0
	dig_λ = digamma(λ)
	dig_λ_sum = digamma(sum(λ))
	for k in 1:m.K
		likelihood += sum(freqs .* φ[:,k] .* (dig_λ[k] .- dig_λ_sum .+ means[k,:] .- m.zeta[k,t_doc] - log(φ[:,k])))
	end
	likelihood += sum(lgamma(λ)) - lgamma(sum(λ)) + sum( (m.alpha .- λ) .* (dig_λ .- dig_λ_sum))
	likelihood
end


"""
Do a heldout perplexity test. Learn topic proportions (given an optimized model) on the first half of a document, then compute the second half likelihood.
Report the per-word predictive perplexity on the second half.
"""
function test(m::GDTM)
	test_ll = 0
	#set a seed, so that we are always looking at the same document parts
	srand(12345)
	total_token_count = 0.
	for doc_id in m.test_doc_ids
		test_doc = m.corpus.documents[doc_id]
		t_doc = DynamicData.get_timestamp_index(m.corpus.documents, test_doc.timestamp)
		(test_doc_words, test_freqs) = CorpusUtils.get_words_and_counts(test_doc.value)
		#expand tokens
		test_doc_tokens = Int64[]
		for (i,w) in enumerate(test_doc_words)
			[push!(test_doc_tokens, w) for j in 1:test_freqs[i]]
		end
		n = length(test_doc_tokens)
		assert(n==sum(test_freqs))
		#split document
		r = randperm(n)
		n_2 = round(Int64, n/2)
		w_1 = test_doc_tokens[r][1:n_2]
		tf_1 = ones(Int64, n_2)
		w_2 = test_doc_tokens[r][n_2+1:end]
		tf_2 = ones(Int64, length(w_2))
		# find optimal topic distribution
		(_, theta, _) = document_inference(m, t_doc, w_1, tf_1)
		#compute phis for second half
		means = zeros(m.K, length(w_2))
		for k in 1:m.K
			means[k,:] = m.KnmKmmInv[t_doc,:]'*m.μ[k][:, w_2]
		end

		phi = zeros(length(w_2), m.K)
		dig_λ = digamma(theta)
		dig_λ_sum = digamma(sum(theta))
		for i in 1:length(w_2)
			log_phi_sum = 0
			for k in 1:m.K
				phi[i,k] = (dig_λ[k] - dig_λ_sum) + (means[k, i] - m.zeta[k, t_doc])

				log_phi_sum = k==1 ? phi[i,k] : log_add(phi[i,k], log_phi_sum)
			end
			#normalize and exp phi
			phi[i,:] = exp(phi[i,:] - log_phi_sum) + 1e-100
		end
		doc_ll = compute_document_likelihood(m, t_doc, w_2, tf_2, phi, theta, means)
		test_ll += doc_ll
		total_token_count += length(w_2)
	end
	-exp(test_ll/total_token_count)
end

"""
Performs a grid search over the parameter space for various kernels (Brownian motion, Cauchy, Ornstein-Uhlenbeck, RBF),
runs the corresponding models and stores the results.
"""
function run_param_gridsearch(batch_size, m_0, s_0, s_x, corpus; visualize=false, num_inducing_points=3, use_seeding=false)
	# different numer of topics
	dims = [5,10]
	# for experiments, alpha is set .5
	alphas = [.5]

	inference_function = (m) -> inference_svi_gp(m, num_inducing_points)

	for alpha in alphas, dim in dims
		m = GDTM(corpus, dim, batch_size, m_0, s_0, s_x, alpha, visualize, seeded=use_seeding)
		range = m.times[end] - m.times[1]
		# effective pre-factor for maximum variance of one in the Brownian motion kernel
		suggested_variance = 1./range
		# different variance parameters for Brownian motion
		wiener_params_mult = [.5, 1., 2.]
		# keep track of best variance in terms of perplexity
		best_nu = 0.
		best_nu_ll = 1e100
		for nu in wiener_params_mult
			m.krn = Kernels.BrownianKernel(nu*suggested_variance)
			inference_function(m)
			write_stats(m, base_dir_suffix="BM")
			if best_nu_ll > m.test_perplexities[end]
				best_nu_ll = m.test_perplexities[end]
				best_nu = nu
			end
			reset(m)
		end
		# minimum lengthscale is equal to space between inducing points
		min_lengthscale = range/num_inducing_points
		lengthscales = [2*range, range, .5*range, .1*range, .05*range]

		for l in lengthscales
			# make sure that maximum variance is as suggested
			m.krn = Kernels.CauchyKernel(best_nu, l)
			inference_function(m)
			write_stats(m, base_dir_suffix="Cauchy")
			reset(m)
		end

		for l in lengthscales
			m.krn = Kernels.OrnsteinUhlenbeckKernel(best_nu, l)
			inference_function(m)
			write_stats(m, base_dir_suffix="OU")
			reset(m)
		end
		for l in lengthscales
			m.krn = Kernels.RBFKernel(best_nu, l)
			inference_function(m)
			write_stats(m, base_dir_suffix="RBF")
			reset(m)
		end
	end
end


"""
Runs a single model based on the corpus, kernel, number of topics, Dirichlet hyperparameter alpha and optional parameters provided.
Stores the result in the current directory under subdirectory "experiments" and creates topic visualizations.
The optional parameters are:
* prior_mean, prior_variance: prior mean and variance for topics
* measurement_noise: measurement noise when drawing time marginals from the stochastic process
* minibatch_size: mini batch size for stochastic algorithm
* inducing_points: number of inducing points for sparse GP
* visualize: if set to true, create probability trajectory charts for top words in topics, report learning rate, e-step (based on the mini batch) and m-step (global optimization step) log likelihoods and test set log likelihoods graphically
* use_seeding: use random documents to seed topics, for each topic at each time step, words in a random document (with appropriate time stamp) are used to artificially increase the probabilities of them in the current topic
"""
function run_model(corpus::CorpusUtils.Corpus, kernel::Kernels.Kernel, num_topics::Int64, alpha::Float64; prior_mean::Float64 = 0., prior_variance::Float64 = 10.,  measurement_noise::Foat64 = .5, minibatch_size::Int64 = 256, inducing_points::Int64 = 25, visualize::Bool = false, use_seeding::Bool = false)
	m = GDTM(corpus, num_topics, batch_size, prior_mean, prior_variance, measurement_noise, alpha, visualize, seeded=use_seeding)
	m.krn = kernel
	GPDTM.gpdtm_svi_inference(m, inducing_points)
	out_dir = GPDTM.write_stats(m)
	if visualize
		GPDTM.make_topic_charts(out_dir, corpus, true, false)
	end
end


"""
Runs all three kernels on the provided dataset for different parameter settings.
The data is expected in the following format:
#{number of timestamps}
#{seconds since epoch, i.e. unix time time stamp 1}
#{number of documents in timestamp 1}
#{types in document 1} #{type id}:#{frequency} #{type id}:#{frequency} ...
...
#{types in document N_1} #{type id}:#{frequency} #{type id}:#{frequency} ...
#{seconds since epoch, i.e. unix time time stamp 2}
#{number of documents in timestamp 2}
#{types in document 1} #{type id}:#{frequency} #{type id}:#{frequency} ...
...
#{types in document N_2} #{type id}:#{frequency} #{type id}:#{frequency} ...
#{repeat for all timestamps}
"""
function main(corpus::CorpusUtils.Corpus; minibatch_size::Int64 = 256, prior_mean::Float64 = 0., prior_variance::Float64 = 10., measurement_noise::Foat64 = .5, inducing_points::Int64 = 25)
	run_param_gridsearch(minibatch_size, prior_mean, prior_variance, measurement_noise, corpus, num_inducing_points=inducing_points)
end

end # module
