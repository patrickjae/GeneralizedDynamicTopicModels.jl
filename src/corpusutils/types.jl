type Word
	id::Int64
	value::AbstractString
	freq::Int64
	doc_freq::Int64
	used_in_docs::Vector{Any}
	function Word(id, value)
		new(id, value, 0, 0, Vector{Any}())
	end
end

typealias Document Dict{Word,Int64}
typealias DocumentList Vector{Document}
typealias DateList Vector{DateTime}

type Lexicon
	words::Vector{Word}
	string_dict::Dict{AbstractString, Int64}
	function Lexicon()
		new(Vector{Word}(), Dict{String, Word}())
	end
	function Lexicon{T<:AbstractString}(strings::Vector{T})
		l = Lexicon()
		for (i, s) in enumerate(strings)
			w = Word(i, s)
			push!(l.words,w)
			l.string_dict[s] = i
		end
		l
	end
end

type Corpus{T<:Union{DocumentList, DynamicData.DynamicDataSet{Document, DateTime}}}
	documents::T
	lexicon::Lexicon

	function Corpus()
		new(T(), CorpusUtils.Lexicon())
	end

	function Corpus{T<:Union{DocumentList, DynamicData.DynamicDataSet{Document, DateTime}}(docs::Union{DocumentList, DynamicData.DynamicDataSet{Document, DateTime}}, lex::Lexicon)
		new(docs, lex)
	end
end

typealias DynamicCorpus Corpus{DynamicData.DynamicDataSet{Document, DateTime}}
