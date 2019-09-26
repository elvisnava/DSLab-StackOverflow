# Paper Summaries

# CQA

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

## ACM_2015_Predicting Answering Behaviour in Online Question Answering Communities

The authors aim to rank the questions by how likely they are to be chosen by a user to anwer it. In contrast to other methods, they include loads of features in their model including detailed information about the question, the user and the whole thread.

First, they construct the list of features extraction user-features (answer reputation, number of posts etc), question (question age, readability scores, number of words etc)- and thread-features (answers that are already there). They train and evaluate three different approaches to predict the question that the user decided to answer, namely Random Forests as a pointwise method, LambdaRank as a pairwise approach, and ListNet for a listwise method.

They evaluated their methods on the Stackexchange Cooking Q&A community. For each user at each time t, they considered all open and available questions as possible candidates for users to select to answer, which was 328 questions per user per time. The random forest performed best, achieving a MRR= 0.446 meaning that selected questions are found on average in the 2nd or 3rd position. They further analyzed which features were most important and found that question features seem most informative, followed by user features. They also individual features by dropping them one by one and computing the accuracy of the random forest. Referal count of the question, question reputation and number of answers were ranked highest. Restricting the features to a subset containing the ones that were highest ranked they could improve the overall accuracy, indication that feature selection can increase performance and reduce computational effort at the same time. 

# Fairness
## NIPS 2017 Beyond Parity
* collaborative filtering framework (i.e. you have a partially observed matrix with users x items and each cell contains a rating [1..5] or unkown)
* usual assumption is that it is random which values are missing. they argue that missing values follow a non-random pattern which is a potential source of unfairness 
* they use a Recomender System for Uni courses as an example and point out that there is a gender imbalance in STEM fields. there are two forms of underrepresentation:
    * **population imbalance** i.e. there are men (M) and Women (W) and they either succeed in Stem (S) or not (N). this defines a 2x2 matrix: due to society WS < WN but MS > MN || this would reflect in Men systematically rating STEM courses higher then Women
    * **observation bias** concerned with which matrix entries are observed. i.e. due to feedback loop women are never recommended STEM courses and thus there are fewer observed entries there
* they introduce 4 fairness metrics (listed at the bottom in general Fairness concepts) and just add them to the loss
* synthetic data
    * each user is part of exactly one group and each item is part of exactly one group
    * for each user type and each item type there is a parameter giving the probability of a user of the type liking an item of that type (population imbalance if nonuniform)
    * for each user type and each item type there is a parameter with the probability of this user-type rating this item-type at all. (observation bias)
* specifically there are 4 user types (WS WN MS MN) and three item types (Fem, STEM, Masc), Masc are 'masculine' courses ;)
* then they have the parameter matrices for this stuff

* there are user/item embeddings and the predicted rating is the inner product, loss is MSE + Frobenius (+ unfairness panalty)
* look at results in paper, I don't yet have intuitive understanding
* when adding (some of the) unfairness terms the overall error actually decreased

* they also do real Movielens data (with genders for users and genres as item groups)

* own idea: by adding the fairness penalty we also introduce information about group identity to the optimisation process. how does it compare if we give that information explicitly to the classifier. e.g. as group based bias or something like that. 






## General Fairness concepts
*  cited as realted work NIPS 2017 Beyond Parity
    * **demographic parity** e.g. binary classification hire/fire P(hire|male) = P(hire|female)
    * **equal opportunity** rougly: accuracy should be same in all groups: e.g. P(hire| male, actual_label=hire) = P(hire | female, actual_label=hire)
* introduced by NIPS 2017 Beyond Parity **all of these actually are also a function of the group I don't know what they do with this** 
    * **value unfairness** eq 6. model over (or under) estimates the rating in one class while under (or over) estimating in the other class. 0 If direction of error is the same in both classes 
    * **absolute unfairness** eq 8. how different the absolut error is in the two classes. does not care about direction. i.e. overestimating by 0.5 in group A and overest. by 1 in group B is the same as overestimating by 0.5 in A and understimating by 1 in group B
    * **underestimation unfairness** checks in each group how much the model underestimates (i.e. cutoff if model overestimates) and computes difference. 
    * **overestimation unfairness** difference in ammount by which the model overestimates (cutoff if it underestimates)
    * **non-parity** difference in average prediction between groups
