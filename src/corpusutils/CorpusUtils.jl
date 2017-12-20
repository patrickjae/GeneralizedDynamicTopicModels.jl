module CorpusUtils
using DataStructures, DynamicData

include("./types.jl")
#Lexicon basic methods
Base.length(l::CorpusUtils.Lexicon) = Base.length(l.words)

function addword(l::CorpusUtils.Lexicon, s::AbstractString, allow_growing = true)
	if haskey(l.string_dict, s)
		w = l.words[l.string_dict[s]]
		w.freq +=1
		return w
	end
	new_word = Word(length(l)+1, s)
	push!(l.words, new_word)
	l.string_dict[s] = new_word.id
	new_word.freq += 1
	new_word
end

function add_text_file(c::CorpusUtils.Corpus, filename::String, timestamp::DateTime; stop_word_file::String="data/stopwords/stop_snowball.txt")
	lines = readlines(filename)
	content = ""
	for line in lines
		content *= line
	end
	d = Document()
	content = replace(content, r"-.?\n", "")
	stopwords = String[]
	try
		stopwords = read_stop_word_file(stop_word_file)
	catch end
	try
		for line in split(content, '\n')
			words = split(line, ' ')
			for word in words
				word = strip(word)
				word = replace(word,r"[[:punct:]]","")
				word = replace(word,['\n','\t']," ")
				word = lowercase(word)
				if word == "" || length(word) ≤ 3 || !isalpha(word) || word ∈ stopwords
					continue
				end
				word_obj = addword(c.lexicon, word)
				if haskey(d, word_obj)
					d[word_obj] += 1
				else
					d[word_obj] = 1
				end
				# println("$word ($(length(word)), $(length(strip(word))))")
			end
		end
	catch UnicodeError
		println("current word is no unicode")
	end
	dp = DynamicData.add(c.documents, d, timestamp)
	for w_obj in collect(keys(d))
		w_obj.doc_freq += 1
		push!(w_obj.used_in_docs, dp)
	end
end

function add_text_file_directory(c::Corpus, path::String, timestamp::DateTime; stop_word_file::String="data/stopwords/stop_snowball.txt")
	for file in readdir(path)
		cur_path = joinpath(path, file)
		if isfile(cur_path)
			add_text_file(c, cur_path, timestamp, stop_word_file=stop_word_file)
		end
	end
end

function read_stop_word_file(filename::String)
	stopwords = String[]
	f = open(filename, "r")
	while !eof(f)
		cur_line = readline(f)
		cur_line = chomp(cur_line)
		if length(cur_line) == 0
			continue
		end
		w = strip(split(cur_line, "|")[1])
		w = replace(w,r"[[:punct:]]","")
		w = replace(w,['\n','\t']," ")
		w = lowercase(w)

		push!(stopwords, w)
	end
	stopwords = collect(filter(w->length(w) != 0, stopwords))
	close(f)
	stopwords
end


#pruning methods
function prune_dictionary!(c::CorpusUtils.Corpus, remove_crit::Function, remove_words_only::Bool=false)
	words_exclude = filter(remove_crit, c.lexicon.words)
	println("removing $(length(words_exclude)) words from corpus")
	c.lexicon.words = setdiff(c.lexicon.words, words_exclude)
	# c.lexicon.words = words_include
	empty!(c.lexicon.string_dict)
	for (id, word) in enumerate(c.lexicon.words)
		c.lexicon.string_dict[word.value] = id
		word.id = id
	end
	if !remove_words_only
		for w in words_exclude
			[delete!(doc.value,w) for doc in w.used_in_docs]
		end
		prune_documents!(c, 1)
	end
end

function prune_documents!(corpus::Corpus, minimum_length::Int64=10)
	docs_remove = filter(x->length(x.value) < minimum_length, corpus.documents)
	# docs_remove = setdiff(collect(1:length(corpus.documents)), docs_keep)
	println("removing $(length(collect(docs_remove))) documents")
	for (i, doc) in enumerate(docs_remove)
		# t_doc = DynamicData.get_timestamp_index(corpus.documents, corpus.documents[d_id].timestamp)
		(words, freqs) = get_words_and_counts(doc.value)
		#first update word stats for each of the words in the document
		for (wi, w) in enumerate(words)
			corpus.lexicon.words[w].freq -= freqs[wi]
			corpus.lexicon.words[w].doc_freq -= 1
			corpus.lexicon.words[w].used_in_docs = setdiff(corpus.lexicon.words[w].used_in_docs, [doc])
			# deleteat!(, findin(corpus.lexicon.words[w].used_in_docs, [doc])[1])
		end
		#remove doc
		DynamicData.remove!(corpus.documents, doc)
	end
	#finally check if we have zero freq words
	prune_unused_words!(corpus)
end

function prune_stop_words!(c::Corpus, stop_words_file)
	words = read_stop_word_file(stop_words_file)
	prune_dictionary!(c, w -> w.value ∈ words)
end

function prune_word_doc_ratio!(c::Corpus, max_doc_ratio::Float64)
	all_docs = length(c.documents)
	prune_dictionary!(c, w -> (length(w.used_in_docs)/all_docs) > max_doc_ratio)
end

prune_word_length!(c::Corpus, min_length::Int64) = prune_dictionary!(c, w -> length(w.value) < min_length)
prune_word_frequency!(c::Corpus, min_freq::Int64) = prune_dictionary!(c, w -> w.freq < min_freq)
prune_word_term!(c::Corpus, term::String) = prune_dictionary!(c, w -> w.value == term)
prune_unused_words!(c::Corpus) = prune_dictionary!(c, w -> w.freq < 1, true)
prune_sparse_words!(c::Corpus, ratio::Float64) = prune_dictionary!(c, w -> ratio > length(w.used_in_times)/length(DynamicData.get_unique_timestamps(c.documents)))

function prune_words_tfidf!(c::Corpus, score_threshold::Float64 = .05)
	all_words = sum(map(x->x.freq, c.lexicon.words))
	all_docs = length(c.documents)
	scores = Dict{Any, Float64}()
	[scores[w] = w.freq/all_words * log(all_docs/w.doc_freq) for w in c.lexicon.words]
	max_value = maximum(collect(values(scores)))
	prune_dictionary!(c, w -> (w.freq/all_words * log(all_docs/w.doc_freq))/max_value < score_threshold)
end
"""
Partitions the dataset by rounding the timestamps to the provided granularity.
Returns a new DynamicDateSet with rounded timestamps (only works for DateTime timestamps).
"""
function get_partitioning(source::CorpusUtils.Corpus, granularity::Union{Type{Dates.Year}, Type{Dates.Month}, Type{Dates.Week}, Type{Dates.Day}})
	target = DynamicData.DynamicDataSet{Document, DateTime}()
	target_lexicon = Lexicon()
	for w in source.lexicon.words
		w_new = Word(w.id, w.value)
		w_new.freq = w.freq
		w_new.doc_freq = w.doc_freq
		w_new.used_in_docs = Vector{DynamicData.DataPoint{Document, DateTime}}()
		push!(target_lexicon.words, w_new)
	end
	target_lexicon.string_dict = deepcopy(source.lexicon.string_dict)
	for d in source.documents
		dp = DynamicData.add(target, d.value, round(d.timestamp, granularity), false)
		[push!(w.used_in_docs, dp) for w in keys(dp.value)]
	end
	DynamicData.rebuild_indexes(target)
	Corpus{DynamicData.DynamicDataSet{Document, DateTime}}(target, target_lexicon)
end

#i/o

function write_sequential_corpus(corpus, filename_corpus, filename_lexicon)
	all_times = DynamicData.get_unique_timestamps(corpus.documents)
	f = open(filename_corpus,"w")
	@printf(f, "%d\n", length(all_times))
	for ts in all_times
		all_docs_t = DynamicData.get_datapoints_for_timestamp(corpus.documents, ts)
		@printf(f, "%f\n%d\n", Dates.datetime2unix(ts), length(all_docs_t))
		for doc in all_docs_t
			(words, freqs) = get_words_and_counts(doc.value)
			@printf(f, "%d", length(words))
			for (i,w) in enumerate(words)
				@printf(f, " %d:%d", w-1, freqs[i])
			end
			@printf(f,"\n")
		end
	end
	close(f)

	f = open(filename_lexicon, "w")
	for w in corpus.lexicon.words
		@printf(f, "%s\n", w.value)
	end
	close(f)
end

function write_corpus(corpus, filename_corpus, filename_lexicon)
	f = open(filename_corpus, "w")
	for doc in corpus.documents
		(words, freqs) = get_words_and_counts(doc.value)
		@printf(f, "%d", length(words))
		for (i,w) in enumerate(words)
			@printf(f, " %d:%d", w-1, freqs[i])
		end
		@printf(f,"\n")
	end
	close(f)
	f = open(filename_lexicon, "w")
	for w in corpus.lexicon.words
		@printf(f, "%s\n", w.value)
	end
	close(f)
end

#read sequential num corpus
function read_sequential_corpus(f, l::Lexicon)
	stream = open(f,"r")
	numTimes = parse(Int64, readline(stream))
	# assume seconds after epoch

	# DynamicData
	c = DynamicData.DynamicDataSet{Document, DateTime}()
	for t=1:numTimes
		cur_date = toDate(readline(stream))
		# cur_date.num_val = t
		numDocsT = parse(Int64,readline(stream))
		for i in 1:numDocsT
			doc = Document()
			#get a doc id
			dp = DynamicData.add(c, doc, cur_date, false)
			t_id = DynamicData.get_timestamp_index(c, cur_date)
			#add the content
			addToDocument(dp, split(readline(stream), "#")[1], l, t_id)
		end
	end
	close(stream)
	DynamicData.rebuild_indexes(c)
	c
end

function toDate(line::AbstractString)
	Dates.unix2datetime(round(Int64, parse(Float64, chomp(line))))
end

function addToDocument(doc::DynamicData.DataPoint, line::AbstractString, l::Lexicon, t_id::Int64)
	for term in split(chomp(line))[2:end]
		(w, doc_freq) = splitTermFreq(term, l)
		push!(w.used_in_docs, doc)
		push!(doc.value, w => doc_freq)
	end
end

function splitTermFreq(term::AbstractString, l::Lexicon)
	parts = split(term, ":")
	w = l.words[parse(Int64, parts[1])+1]
	doc_freq = parse(Int64,parts[2])
	w.freq += doc_freq
	w.doc_freq += 1
	Pair(w, doc_freq)
end

#read lexicon
function read_lexicon(stream)
	lines = readlines(stream)
	Lexicon(map(chomp, lines))
end


function get_words_and_counts(doc::Document)
	(map(w->w.id, collect(keys(doc))), collect(values(doc)))
end


function create_corpus(lexicon_file::String, corpus_file::String)
	l = CorpusUtils.read_lexicon(lexicon_file)
	c = CorpusUtils.read_sequential_corpus(corpus_file, l)
	CorpusUtils.Corpus(c,l)
end

function used_in_times(c::Corpus, w::Word)
	DynamicData.get_timestamp_indexes_for_subset(c.documents, convert(Vector{DynamicData.DataPoint{Document, DateTime}}, w.used_in_docs))
end
end # module
