# GeneralizedDynamicTopicModels.jl
GeneralizedDynamicTopicModels allow for arbitrary Gaussian Process (GP) priors on the drift of topics in Dynamic Topic Models as described by [1] (which only allows for a Brownian motion prior).
Further, it introduces a scalable inference algorithm based on Stochastic Variational Inference for GPs based on [2].
This package accompanies our AISTATS2018 paper [3] and is written in Julia.

# Installation
Installation should work with the Julia-integrated package manager. 
From the Julia REPL, simply issue the command
```
Pkg.clone("https://github.com/patrickjae/GeneralizedDynamicTopicModels.jl")
```
and you should be good to go.
```using GeneralizedDynamicTopicModels```imports the necessary modules (GeneralizedDynamicTopicModels, CorpusUtils, DynamicData)

# Usage
For using the code, you first have to have access to a timestamped document collection.
The CorpusUtils module provides data structures Corpus, Document and Word to construct a document collection.
We give two automated options to use your data.

## Importing preprocessed data
In case you want to use your own preprocessing pipeline, you need to provide the data in a variant of the c-LDA format.
The command
```
CorpusUtils.create_corpus(lexicon_file, corpus_file)
```
creates a corpus object on the basis of the data and a lexicon file that are structure as follows:

### Lexicon file
The lexicon file contains the vocabulary used in the corpus. The line number encodes the internal ID of the word type (0-based indexing).

### Corpus file
The corpus file can be constructed by
1. print the number of timestamps $T$ in the corpus as an integer value _line break_
2. for each timestamp ![](http://latex.codecogs.com/svg.latex?t\in\{1,\ldots,T\}) :
	1. print timestamp (in number of seconds since 1.1.1970, may be negative) _line break_
	2. print number of documents ![](http://latex.codecogs.com/svg.latex?D_t) in timestamp ![](http://latex.codecogs.com/svg.latex?t) _line break_
	3. for each document ![](http://latex.codecogs.com/svg.latex?d\in\{1,\ldots,D_t\}) :
		1. print number of unique word types ![](http://latex.codecogs.com/svg.latex?N_d) in document ![](http://latex.codecogs.com/svg.latex?d) 
		2. for each word type ![](http://latex.codecogs.com/svg.latex?w_n), print _space_![](http://latex.codecogs.com/svg.latex?w_n):![](http://latex.codecogs.com/svg.latex?\text{freq}(w_n))
		3. print _line break_

![](http://latex.codecogs.com/svg.latex?w_n) denotes the ID of the word type, i.e. the line number (-1) in the lexicon file.

A sample corpus file could look like
```
2
1.513772177301658e9
3
2 0:5 3:3
5 1:3 0:1 3:1 2:2 4:1
3 0:5 2:3
1.513772177301662e9
2
4 0:2 5:4 1:1 4:1
3 1:4 4:2 5:2
```
with associated lexicon file
```
word 1
word 2
word 3
word 4
word 5
word 6
```

## Importing plain text data
You can import single plain text files (with a timestamp) or complete directories of plain text files (all sharing a common timestamp).
For this, first create an empty Corpus object:
```
corpus = CorpusUtils.DynamicCorpus()
```
For each file you want to import into the corpus, call the add_text_file method in the CorpusUtils module
```
CorpusUtils.add_text_file(corpus, filename, timestamp, stop_word_file="stop_word_file_location")
```
where _filename_ points to the file to import, _timestamp_ is a DateTime object and _stop_word_file_ is an optional list of stopwords (one on each line).
Using this command, there are a few implicit preprocessing steps:
- all words are lowercased
- punctation is removed
- new lines and tabs are replaced by a space
- words of length 3 or below are omitted
- numbers are discarded
- if a stop words file is provided, stop words are omitted as well

For complete directories of plain text files all sharing the same timestamp use
```
CorpusUtils.add_text_file_directory(corpus, path, timestamp, stop_word_file="stop_word_file_location")
```
where now path points to the directory containing the data. Other parameters are as before.

Both options allow for further pruning of the dictionary after importing the data:
1. ```CorpusUtils.prune_word_length!(corpus, minimum_length)``` removes any words from the corpus that are shorter then the minimum length
2. ```CorpusUtils.prune_word_frequency!(corpus, minimum_frequency)``` removes any words with frequency less than minimum across the corpus
3. ```CorpusUtils.prune_word_term!(corpus, term)``` removes a specific word type from the corpus
4. ```CorpusUtils.prune_documents!(corpus, min_length)``` removes documents with effective length (i.e. all word type frequencies summed) less than the minimum
5. ```CorpusUtils.prune_unused_words!(corpus)``` removes words not used (can happen when removing documents)
6. ```CorpusUtils.prune_words_tfidf!(corpus, threshold)``` computes a corpus wide tf-idf score for each word and removes all words with score smaller than threshold

You can define your own rules by calling the function ```CorpusUtils.prune_dictionary!(corpus, function)``` where ```function``` takes a ```CorpusUtils.Word``` object and must return true if the word should be deleted from the corpus.

After having done all preprocessing steps, you can store your corpus in dynamic c-LDA format using
```CorpusUtils.write_corpus(corpus, corpus_file, lexicon_file)```
for further usage.

To read in a corpus saved this way, use the procedure as described above.

## Running the model
Having loaded the corpus data, the model can be run in different ways.

### Doing a parameter grid search
With a call to 
```
GeneralizedDynamicTopicModels.main(corpus)
``` 
the model can be run directly. Additional parameters that can be used include.

- prior_mean, prior_variance: the topic prior hyperparameters, default: 0.0, 10.0
- measurement_noise: the variance used when drawing words from the time marginal of the process, default: 0.5
- minibatch_size: parameter for the stochastic algorithm, default: 256
- inducing_points: number of inducing points used for the sparse GP approximation, default: 25

This method uses four different kernels, corresponding to different covariance functions (see below)
- Brownian motion
- Cauchy
- Ornstein-Uhlenbeck
- RBF

and computes models for different parameter settings and kernel hyperparameters.
Results are stored in subdirectories of "experiments", e.g. "experiments/BM". Probability trajectories of top words in topics can be created using a call to
```GeneralizedDynamicTopicModels.make_all_charts(path, corpus, save_images)```
where path should point to one of the created subdirectories of "experiments", e.g. "experiments/BM".
The boolean "save_images" determines whether the charts are saved as PDF files. Charts for training and test set ELBO are always created.
An optional keyword parameter "show_images" can be added to the function call with 
```GeneralizedDynamicTopicModels.make_all_charts(path, corpus, save_images, show_images=true)```
to display the charts directly.


### Running a single model
If you want to do your own set of experiments, use a call to
```GeneralizedDynamicTopicModels.run_model(corpus, kernel, num_topics, alpha)```
and provide the corpus, a kernel object (see below), the number of topics to use and the alpha hyperparameter to the Dirichlet prior to the topic proportions of a document.

Additional keyword parameters are 
- visualize: do a visualization right away, including trajectories for learning rate, minibatch log likelihood (scaled), global step log likelihood and test set predictive log likelihood.
- use_seeding: for each topic at each timestamp, use a random document (of that timestamp) to artifically increase the probability of its words in that topic, other topic-word probabilites are generated from the prior
- all keyword parameters used in the grid search above


## Kernels
The Kernels module provides different kernels that are used to define the stochastic process prior over time. 
W.l.o.g. different GP priors are fully defined by different kernel (or covariance) functions ![](http://latex.codecogs.com/svg.latex?\kappa(x,x^\prime)).
Up to now, the following kernels have been implemented:
1. Brownian Kernel ```Kernels.BrownianKernel``` with process variance parameter ![](http://latex.codecogs.com/svg.latex?\sigma^2), ![](http://latex.codecogs.com/svg.latex?\kappa(x,x^\prime)=\sigma^2\min(x,x^\prime))
2. Ornstein-Uhlenbeck Kernel ```Kernels.OrnsteinUhlenbeckKernel``` with process variance parameter ![](http://latex.codecogs.com/svg.latex?\sigma^2) and length scale ![](http://latex.codecogs.com/svg.latex?\ell), ![](http://latex.codecogs.com/svg.latex?\kappa(x,x^\prime)=\sigma^2\exp\left(-\frac{||x-x^\prime||}{\ell}\right))
3. Cauchy Kernel ```Kernels.CauchyKernel``` with process variance parameter ![](http://latex.codecogs.com/svg.latex?\sigma^2) and length scale ![](http://latex.codecogs.com/svg.latex?\ell), ![](http://latex.codecogs.com/svg.latex?\kappa(x,x^\prime)=\frac{\sigma^2}{1+\left(\frac{||x-x^\prime||}{\ell}\right)^2})
2. RBF Kernel ```Kernels.RBFKernel``` with process variance parameter ![](http://latex.codecogs.com/svg.latex?\sigma^2) and length scale ![](http://latex.codecogs.com/svg.latex?\ell), ![](http://latex.codecogs.com/svg.latex?\kappa(x,x^\prime)=\sigma^2\exp\left(-\frac{||x-x^\prime||^2}{2\ell^2}\right))
4. Rational Quadratic Kernl ```Kernels.RationalQuadraticKernel``` with process variance parameter ![](http://latex.codecogs.com/svg.latex?\sigma^2), length scale ![](http://latex.codecogs.com/svg.latex?\ell) and alpha parameter ![](http://latex.codecogs.com/svg.latex?\alpha) (governing the weighting of large- and small-scale variations), ![](http://latex.codecogs.com/svg.latex?\kappa(x,x^\prime)=\sigma^2\left(1+\left(\frac{||x-x^\prime||^2}{2\alpha\ell^2}\right)^{-\alpha}\right))
5. Constant Kernel ```Kernels.ConstantKernel``` with process variance parameter ![](http://latex.codecogs.com/svg.latex?\sigma^2) (which is used to populate the whole covariance matrix), ![](http://latex.codecogs.com/svg.latex?\kappa(x,x^\prime)=\sigma^2)
6. Periodic Kernel ```Kernels.PeriodicRBFKernel``` with process variance parameter ![](http://latex.codecogs.com/svg.latex?\sigma^2), length scale parameter ![](http://latex.codecogs.com/svg.latex?\ell) and period parameter ![](http://latex.codecogs.com/svg.latex?p), ![](http://latex.codecogs.com/svg.latex?\kappa(x,x^\prime)=\sigma^2\exp\left(-2\frac{\sin\left(\frac{||x-x^\prime||}{p}\right)^2}{\ell^2}\right))

Note that we have not yet conducted exhaustive experiments with the rational quadratic and the periodic kernels. 
Especially the latter has show to be extremely instable and thus probably needs a fair amount of parameter tuning.
Additional kernels can be simply added by defining a type for the kernel that is a subtype of ```Kernels.Kernel```, e.g.
```
type MyFancyKernel <: Kernel
	parameters::Vector{Float64}
	MyFancyKernel(param1::Float64, param2::Float64) = new([param1, param2])
end
```
and implementing the method ```Kernels.computeCovariance(kernel, x, x_prime)``` and explicitly providing type information on the kernel (i.e. the type create for this kernel). ```x``` and ```x_prime``` are vectors of ```Float64``` and the method should return a ```Float64``` value, i.e.
```
computeCovariance(k::MyFancyKernel, x::Vector{Float64}, x_prime::Vector{Float64}) = some function of x, x_prime and k.parameters
```


## References

[1] Wang, C., Blei, D. M., & Heckerman, D. (2008). Continuous Time Dynamic Topic Models. UAI

[2] Hensman, J., Fusi, N., & Lawrence, N. D. (2013). Gaussian Processes for Big Data. UAI.

[3] Jähnichen, P., Wenzel, F., Kloft, M., & Mandt, S. (2018). Scalable Generalized Dynamic Topic Models. AISTATS.
