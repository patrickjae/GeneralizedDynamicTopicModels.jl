function inference_svi_gp(m::GDTM, num_inducing; rand_inducing::Bool=false, normalize_timestamps::Bool=false)
	tic()

	e_step_likelihoods = Float64[]
	m_step_likelihoods = Float64[]
	full_elbo = Float64[]

	test_schedule = 1
	cur_count = 0

	# size of the minibatch used to compute updates and samples
	mini_batch_size = m.batch_size

	# set the number of epochs (needs some kind of manipulation, monitor elbo, map ?)
	# println("handling $(m.D) docs")
	epochs = round(Int64, m.D/mini_batch_size)
	info("doing $epochs epochs with minibatch size $mini_batch_size")
	info("parameter:\nKernel: $(m.krn)\nalpha: $(m.alpha)\nK: $(m.K)")

	# timestamp normalization is disabled by default
	if normalize_timestamps
		m.times .-= minimum(m.times) - 1
		m.times = m.times ./ maximum(m.times)
	end

	# random inducing point selection is disabled by default
	# with a roughly uniform distribution of documents over timestamps, even spacing works better
	if num_inducing >= m.T
		# if number of inducing points is larger than total timestamps, use all (degenerating into a full rank model)
		m.inducing_points = m.times
		num_inducing = m.T
	elseif rand_inducing
		# randomly select induing points from time points
		m.inducing_points = m.times[randperm(m.T)][1:num_inducing]
	else
		m.inducing_points = collect(linspace(minimum(m.times),maximum(m.times), num_inducing))
	end
	# compute number of all tokens in the training set, used for likelihood smoothing
	all_tokens = sum(map(d->sum(collect(values(d.value))), m.corpus.documents[m.training_doc_ids]))

	# compute inducing points covariance matrix
	m.Kmm = Hermitian(Kernels.computeCovarianceMatrix(m.krn, m.inducing_points))

	m.jitter = Diagonal(ones(num_inducing) .* 1e-7)
	m.KmmInv = inv(Hermitian(m.Kmm + m.jitter))
	jitter_full = Diagonal(ones(m.T) .* 1e-7)

	# compute training covariance matrix
	m.Knn = Hermitian(Kernels.computeCovarianceMatrix(m.krn, m.times) + eye(m.T).*m.s_x + jitter_full)

	# compute cross covariance and some intermediate results
	m.Knm = Kernels.computeCrossCovarianceMatrix(m.krn, m.times, m.inducing_points)
	m.KnmKmmInv = m.Knm*m.KmmInv

	m.K_tilde = Hermitian(m.Knn - m.KnmKmmInv*m.Knm')
	m.K_tilde_diag = diag(m.K_tilde)

	# optionally plot heatmaps of covariance matrices for debugging
	if m.visualize
		display(Plots.heatmap(m.Knn, title="original covariance"))
		display(Plots.heatmap(m.Knm*m.KnmKmmInv', title="reconstructed covariance"))
		display(Plots.heatmap(m.Kmm, title="inducing point covariance"))
		display(Plots.heatmap(m.K_tilde, title="reconstruction error"))
	end

	info("initializing the model...")

	# initialize means, covariances and natural parameters
	m.μ = Vector{Matrix{Float64}}(m.K)
	m.s = Matrix{Matrix{Float64}}(m.K, m.V)
	m.eta_1 = Vector{Matrix{Float64}}(m.K)
	m.eta_2 = Matrix{Matrix{Float64}}(m.K, m.V)

	L = chol(m.Knn)'

	# for mean initialization, use document seeding (one doc per topic and time) or draw randomly from a GP with covariance as given by the timestamps
	if m.use_seeding
		Threads.@threads for t in 1:m.T
			all_docs_t = DynamicData.get_datapoints_for_timestamp(m.corpus.documents, m.timestamps[t])
			p = zeros(Int64, m.K)
			if length(all_docs_t) < m.K
				for i in 1:m.K
					p[i] = rand(1:length(all_docs_t))
				end
			else
				p = randperm(length(all_docs_t))[1:m.K]
			end
			mean = zeros(m.V)
			for k in 1:m.K
				doc = m.corpus.documents[p[k]].value
				(words, freqs) =  CorpusUtils.get_words_and_counts(doc)
				mean = randn(m.V).*m.s_0 + m.m_0
				mean[words] = log(freqs .+ rand(length(words)))
				m.μ[k] = m.KnmKmmInv[t,:]' \ mean'
			end
		end
	end

	# initialize variational covariances and natural parameters to topic distributions
	@inbounds for k in 1:m.K
		if !m.use_seeding
			m.μ[k] = zeros(num_inducing, m.V)
		end
		m.eta_1[k] = zeros(num_inducing, m.V)

		Threads.@threads for w in 1:m.V
			if !m.use_seeding
				m.μ[k][:,w] = m.KnmKmmInv \ (L*randn(m.T))
			end
			m.s[k,w] = eye(num_inducing)
			m.eta_2[k,w] = -.5*eye(num_inducing)
			m.eta_1[k][:,w] = -2*m.eta_2[k,w] * m.μ[k][:,w]
		end
	end

	# assume s is identity matrix for init
	# compute the Λ diagonals and update 𝜁 parameters
	init_val = diag(m.KnmKmmInv*m.KnmKmmInv')
	Λ_diags = zeros(m.T, m.V)
	for k in 1:m.K
		Threads.@threads for w in 1:m.V
			Λ_diags[:,w] = copy(init_val)
		end
		means = m.KnmKmmInv*m.μ[k]
		svi_update_zeta(m, k, means, Λ_diags)
	end

	# parameters to steer the step size in stochastic gradient updates
	a = .1
	b = 10
	gamma = .7

	# helpers for performing inference loop
 	iter = 0
  	last_e_step_ll = -1e100
  	svi_counter = 1
	mb_idx_rand = m.training_doc_ids[randperm(m.D)]
	e_step_time_agg = 0.
	m_step_time_agg = 0.
	info("done, starting inference")
	@fastmath @inbounds for e in 1:epochs
		tic()

	    iter += 1
		# determine the minibatch to operate on
		end_pos = svi_counter + mini_batch_size - 1
		if end_pos > m.D
			end_pos = m.D
		end
		mb_idx = mb_idx_rand[svi_counter:end_pos]

		# keep track of documents seen
		cur_count += end_pos - svi_counter + 1

		# when reading the end of the training set, reshuffle the documents and start at beginning again
		svi_counter += mini_batch_size
		if svi_counter > m.D
			svi_counter = 1
			mb_idx_rand = m.training_doc_ids[randperm(m.D)]
			iter = 1
		end
		# token count in the minibatch, gives a more realistic estimate of the fraction N/|S| for the gradient (instead of fractions of document counts)
		mb_wordcount = sum([sum(collect(values(doc.value))) for doc in m.corpus.documents[mb_idx]])
		mult = all_tokens/mb_wordcount

		# compute the RM learning rate (using a common form)
	    lr = a*(b+iter)^-gamma
	    # keep track of learning rates
		push!(m.learning_rates, lr)

		#do local udpate step (termed "e-step")
		(online_bound, ss_tk, ss_tkx, words_seen, t_mb) = e_step(m, mb_idx)
		# multiply bound estimate by multiplier
		online_bound *= mult
		# estimate of document specific bound for whole corpus
		push!(e_step_likelihoods, online_bound)

		e_step_time = toq()
		e_step_time_agg += e_step_time

		# m-step, update global variables
		tic()
		# compute natural gradients according to sufficient statistics (gradient w.r.t. to canonical params is natural gradient w.r.t. natural params)
		m_step_bound = 0.
		# only look at words actually observed in the minibatch
		KnmKmmInvEff = m.KnmKmmInv[t_mb,:]

		Λ_diags = zeros(m.T, m.V)
		means = zeros(m.T, m.V)
		word_bounds = zeros(m.V)
		for k in 1:m.K
			ɸ_k = mult.*ss_tk[k][t_mb] #(T x 1)
			Ξ_k = mult.*ss_tkx[k][t_mb, :] #(T x V)
			
			Threads.@threads for w in 1:m.V
				Λ_diags[t_mb,w] = vec(sum(KnmKmmInvEff*m.s[k,w] .* KnmKmmInvEff, 2))
			end

			means[t_mb, :] = KnmKmmInvEff*m.μ[k]

			B_tilde_k = ɸ_k .* exp(means[t_mb,:] .+ .5(Λ_diags[t_mb,:] .+ m.K_tilde_diag[t_mb]) .- m.zeta[k,t_mb])
			# euclidean gradient for mean
			dL_dm = KnmKmmInvEff' * (Ξ_k + B_tilde_k .* (means[t_mb,:] .- 1))

			m.eta_1[k] = (1-lr)*m.eta_1[k] + lr.*dL_dm
			Threads.@threads for w in 1:m.V
				# euclidean gradient for variance
				dL_dS = -.5*(m.KmmInv + (KnmKmmInvEff .* B_tilde_k[:,w])'*KnmKmmInvEff)

				# update step for second natural parameter
				m.eta_2[k,w] = (1-lr)*m.eta_2[k,w] + lr.*dL_dS

				# compute inverse and determinant using cholesky decomposition if possible
				eta_inv, det_eta_inv = chol_inv(m.eta_2[k,w] + m.jitter)
				# compute new covariance matrix and determinant (inducing points)
				m.s[k,w] = -.5*eta_inv
				det_s = (-.5)^num_inducing * det_eta_inv

				# compute new mean (inducing points)
				m.μ[k][:,w] = m.s[k,w]*m.eta_1[k][:,w]

				Λ_diags[:,w] = vec(sum(m.KnmKmmInv*m.s[k,w] .* m.KnmKmmInv,2))
				p_u = -.5*sum(m.KmmInv.*m.s[k,w])
				q_u = -.5*log(abs(det_s))
				word_bounds[w] = p_u-q_u
			end
			# compute means for all timestamps (needed to recompute zeta)
			means = m.KnmKmmInv*m.μ[k]
			m_step_bound += -.5*sum(m.KmmInv*m.μ[k] .* m.μ[k])
			# update zetas again
			svi_update_zeta(m, k, means, Λ_diags)

			m_step_bound += sum(word_bounds)
		end

		push!(m_step_likelihoods, m_step_bound)
		online_bound += m_step_bound
		m_step_time = toq()
		m_step_time_agg += m_step_time
		info("epoch $e, current elbo: $online_bound, e-step: $e_step_time, m-step: $m_step_time")

		# keep track of likelihoods
		push!(m.likelihoods, online_bound)
		push!(m.likelihood_counts, cur_count)

		# test on unseen data according to schedule
		if length(m.test_doc_ids) > 0 && e % test_schedule == 0
			test_ppx = test(m)
			push!(m.test_counts, cur_count)
			push!(m.test_perplexities, test_ppx)
			if test_schedule < 256 test_schedule *= 2	end
		end
	end
	if m.visualize
		display(Plots.plot(m.learning_rates, label="learning rates"))
		display(Plots.plot(m.likelihood_counts, e_step_likelihoods, label="e-steps"))
		display(Plots.plot(m.likelihood_counts, m_step_likelihoods, label="m-steps"))
		display(Plots.plot(m.test_counts, m.test_likelihoods, title="test set predictive likelihood"))
	end
	time = toq()

	info("time for GP SVI: $time seconds, average e-step time: $(e_step_time_agg/epochs), average m-step time: $(m_step_time_agg/epochs)")
	info("e-step full: $e_step_time_agg, m-step full: $m_step_time_agg")
	time
end



function svi_update_zeta(m::GDTM, k::Int64, means::Matrix{Float64}, Λ_diags::Matrix{Float64})
	Threads.@threads for t in 1:m.T
		m.zeta[k,t] = log_sum(means[t,:] + .5*(Λ_diags[t,:] .+ m.K_tilde[t,t]))
	end
end

function svi_update_zeta(m::GDTM)
	for k in 1:m.K
		svi_update_zeta(m, k)
	end
end
