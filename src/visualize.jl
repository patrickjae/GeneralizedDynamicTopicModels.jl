function make_all_charts(parent_directory::String, corpus_file::String, lexicon_file::String, save_images::Bool; show_images::Bool=false)
	if save_images Plots.plotlyjs() elseif show_images Plots.plotly() end
	corpus = CorpusUtils.create_corpus(lexicon_file, corpus_file)
	all_tokens = sum(map(d -> sum(values(d.value)), corpus.documents))
	info("all tokens: $all_tokens")
	all_likelihoods = Plots.plot(title="ELBOs", size=(1024, 768), ylabel="lower bound on marginal likelihood", xlabel="# documents seen")
	all_test = Plots.plot(title="Predictive likelihood", ylabel="predictive likelihood", xlabel="# documents seen", size=(1024, 768), tickfont=font(14, "sans-serif"))
	efn = ""
	ppx_values = Dict{String, Float64}()
	for dir in readdir(parent_directory)
		if isdir(joinpath(parent_directory, dir))
			kernel_params = match(r"\[(.*)\]", dir)[1]
			kernel_name = replace(dir, r"Kernel(Real)?\[.*\]", "")
			for alpha_dir in readdir(joinpath(parent_directory, dir))
				cur_alpha_dir = joinpath(parent_directory, dir, alpha_dir)
				alpha = replace(alpha_dir,"alpha", "")
				if isdir(cur_alpha_dir)
					for model_param_var in readdir(cur_alpha_dir)
						cur_model_param_path = joinpath(cur_alpha_dir, model_param_var)
						if isdir(cur_model_param_path)
							for num_topics in readdir(cur_model_param_path)
								num_topics_path = joinpath(cur_model_param_path, num_topics)
								if isdir(num_topics_path)
									K = parse(Int64, replace(num_topics, "_topics", ""))
									inducing_point_model = false
									if isfile(joinpath(num_topics_path, "inducing_times.dat")) inducing_point_model = true end
							    	println("processing $num_topics_path (is inducing point model: $inducing_point_model)")
							    	# plot_label = "$kernel_name($kernel_params), ⍺=$alpha, K=$K"
							    	my_kernel_name = replace(kernel_name, r"Brownian", "Wiener")
							    	my_kernel_name = replace(my_kernel_name, r"OrnsteinUhlenbeck", "Ornstein Uhlenbeck")
							    	plot_label = "$my_kernel_name kernel"
							    	(c, e) = tab_split(joinpath(num_topics_path, "elbo"))
							    	Plots.plot!(all_likelihoods, c, e, label=plot_label, linewidth=2, legendfont=font(18, "sans-serif"))
							    	(tc, tppx) = tab_split(joinpath(num_topics_path, "test"))
							    	ppx_values[plot_label] = tppx[end]
							    	Plots.plot!(all_test, tc, tppx, label=plot_label, linewidth=2, legendfont=font(18, "sans-serif"))
							    	make_topic_charts(num_topics_path, corpus, save_images, show_images)
							    end
							end
						end
					end
				end
			end					
	    end
	end
	display(ppx_values)
	ppx_file = open(joinpath(parent_directory, "perplexities"), "w")
	for (key, val) in ppx_values
		println("$key: $val")
		@printf(ppx_file, "%s\t%f\n", key, val)
	end
	close(ppx_file)
	if save_images
		Plots.pdf(all_likelihoods, joinpath(parent_directory, "elbos"))
		Plots.pdf(all_test, joinpath(parent_directory, "tests"))
		Plots.plotly()
	else
		display(all_likelihoods)
		display(all_test)
	end
end

function make_topic_charts(path::String, corpus::CorpusUtils.Corpus, save_images::Bool, show_images::Bool)
	if save_images 
		if isdir(joinpath(path, "word_trajectories"))
			return 
		end
		Plots.plotlyjs()
	elseif show_images
		Plots.plotly()
	end
	all_colors = Colors.distinguishable_colors(length(corpus.lexicon))
	all_times = readlines(joinpath(path, "times.dat"))

	ticks = DynamicData.get_unique_timestamps(corpus.documents)

	if length(ticks) < 20
		tick_pos = collect(1:length(ticks))
	else
		tick_pos = collect(1:10:length(ticks))
	end
	tick_labels = String[]

	juliandays = map(Dates.datetime2julian, ticks)
	time_range = maximum(juliandays) - minimum(juliandays)
	println("time range: $time_range")
	if time_range > 21*365
		tick_labels = map(string, map(Dates.year, ticks))
	elseif time_range > 2*365
		tick_labels = map(ts -> Dates.format(ts, "yyyy-mm"), ticks)
	else
		tick_labels = map(ts -> Dates.format(ts, "yyyy-mm-dd"), ticks)
	end
	ticks = tuple(tick_pos, tick_labels[tick_pos])

	topic_probabilities = get_topic_distributions(path, save_images)
	K = length(topic_probabilities)
	for k in 1:K
		(T,V) = size(topic_probabilities[k])
		word_means = vec(mean(topic_probabilities[k],1))
		words_to_plot = sortperm(word_means, rev=true)

		p = Plots.plot(size=(1024, 768), xlabel="Time", ylabel="Probability in topic", xticks=ticks, grid=false, tickfont=font(14, "sans-serif"), guidefont=font(16, "sans-serif"), rotation=90, legend=:legend, color_palette=all_colors[words_to_plot])
		for w in words_to_plot
			for t in 1:T
				if isnan(topic_probabilities[k][t,w]) || isinf(topic_probabilities[k][t,w])
					println("warning, word prob is nan or inf")
					topic_probabilities[k][t,w] = 0.
				end
			end
			f_w = topic_probabilities[k][:,w]
			Plots.plot!(p, collect(1:T), f_w, label=corpus.lexicon.words[w].value, linewidth=4, legendfont=font(20, "sans-serif"))
		end

		if save_images
			img_path = joinpath(path, "word_trajectories")
			mkpath(img_path)
			Plots.pdf(p, joinpath(img_path, "topic$k"))
		elseif show_images 
			display(p)
		end
	end
	if save_images Plots.plotly() end
end
