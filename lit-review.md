# Paper Summaries

## 2010 - Expert Identification in Community Question Answering
The contribution consists of using a simple bias-based model to identify experts in the TurboTax community. Labels for experts are manually assigned by TurboTax moderators.

The bias consists in wether a user tends to answer questions with a "low score", meaning questions with no existing answers or poor quality existing answers. The obtained model is compared with a simple text-based classification model and one based on a simple "Z-score".The bias model obtains very bad precision but very good recall. Combining it with the "Z-score" yields the best F1 score.

The bias model can be modified to obtain higher precision (without affecting recall) by only considering users with more than N answers (however it can be seen from the graph that also the "random" classifier increases its precision in the same fashion, so it's not really about biases inherent in low volume users, rather than the fact that N is a proxy for being an expert). A better way of improving the model is by only considering the first month of answers of each user. This increases both accuracy and recall, and makes the Gaussian Model stand out as the most accurate.

Lastly, the authors divide the data in 5 time slots and obtain the same conclusions for each, meaning that there probably were no outside influences generating this kind of bias phenomenon, but it remained stable during the lifetime of the website.

## 2012 - Finding Expert Users in Community Question Answering
The paper focuses on using topic modeling (LDA and STM) to build user profiles and identify the best experts for answering new questions.

User interests can be inferred with word-based methods and topic models. A standard word-based method is TF-IDF, a measure for the importance of a word in a document based on the frequency of the word in the document and the inverse proportion of all documents containing the word. Given a question and a user (her post history), the recommendation score is obtained using the Cosine Similarity of the tf-idf scores over all the words in the question and post history. Another word-based method is *Language Model*, which uses a multinomial probability distribution over words to represent a user, avoiding zero probabilities with Dirichlet smoothing.

Latent Dirichlet Allocation (LDA) is a hierarchical Bayesian topic model in which each document is modeled as a mixture of topics, each of which is a distribution over words in a vocabulary. Gibbs sampling is used to estimate the paramenters of the model.

Segmented Topic Model (STM) is a topic model that discovers the hierarchical structure of topics by using the two-parameter Poisson Dirichlet process. Each question (segment) in a user history has its own topic distribution.

The approaches are trained on a subset of a StackOverflow dump and evaluated using "success at N" (S@N), meaning that a prediction is considered successful if the model could find the actual best answerer of a question among the top N predicted users. Topic models exhibit much better performance than word-based approaches, and STM performs better than LDA, also extracting more meaningful topics.
