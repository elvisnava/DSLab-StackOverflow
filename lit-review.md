# Paper Summaries

# CQA

## CIKM 2010 - Expert Identification in Community Question Answering
The contribution consists of using a simple bias-based model to identify experts in the TurboTax community. Labels for experts are manually assigned by TurboTax moderators.

The bias consists in whether a user tends to answer questions with a "low score", meaning questions with no existing answers or poor quality existing answers. The obtained model is compared with a simple text-based classification model and one based on a simple "Z-score".The bias model obtains very bad precision but very good recall. Combining it with the "Z-score" yields the best F1 score.

The bias model can be modified to obtain higher precision (without affecting recall) by only considering users with more than N answers (however it can be seen from the graph that also the "random" classifier increases its precision in the same fashion, so it's not really about biases inherent in low volume users, rather than the fact that N is a proxy for being an expert). A better way of improving the model is by only considering the first month of answers of each user. This increases both accuracy and recall, and makes the Gaussian Model stand out as the most accurate.

Lastly, the authors divide the data in 5 time slots and obtain the same conclusions for each, meaning that there probably were no outside influences generating this kind of bias phenomenon, but it remained stable during the lifetime of the website.

**metric** they have a binary classification problem with labeled experts, => precission, recall, F1 etc.

## WWW 2012 - Finding Expert Users in Community Question Answering
The paper focuses on using topic modeling (LDA and STM) to build user profiles and identify the best experts for answering new questions.

User interests can be inferred with word-based methods and topic models. A standard word-based method is TF-IDF, a measure for the importance of a word in a document based on the frequency of the word in the document and the inverse proportion of all documents containing the word. Given a question and a user (her post history), the recommendation score is obtained using the Cosine Similarity of the tf-idf scores over all the words in the question and post history. Another word-based method is *Language Model*, which uses a multinomial probability distribution over words to represent a user, avoiding zero probabilities with Dirichlet smoothing.

Latent Dirichlet Allocation (LDA) is a hierarchical Bayesian topic model in which each document is modeled as a mixture of topics, each of which is a distribution over words in a vocabulary. Gibbs sampling is used to estimate the paramenters of the model.

Segmented Topic Model (STM) is a topic model that discovers the hierarchical structure of topics by using the two-parameter Poisson Dirichlet process. Each question (segment) in a user history has its own topic distribution.

The approaches are trained on a subset of a StackOverflow dump and evaluated using "success at N" (S@N), meaning that a prediction is considered successful if the model could find the actual best answerer of a question among the top N predicted users. Topic models exhibit much better performance than word-based approaches, and STM performs better than LDA, also extracting more meaningful topics.

**metric:** Success at N. See if actuall answerer is in the top N predicted candidates => counts as reciprocal rank (e.g. actual answer was predicted as 3rd out of ten= 7/10) otherwise 0 => average

## ACM_2015_Predicting Answering Behaviour in Online Question Answering Communities

The authors aim to rank the questions by how likely they are to be chosen by a user to answer it. In contrast to other methods, they include loads of features in their model including detailed information about the question, the user and the whole thread.

First, they construct the list of features extraction user-features (answer reputation, number of posts etc), question (question age, readability scores, number of words etc)- and thread-features (answers that are already there). They train and evaluate three different approaches to predict the question that the user decided to answer, namely Random Forests as a pointwise method, LambdaRank as a pairwise approach, and ListNet for a listwise method.

They evaluated their methods on the Stackexchange Cooking Q&A community. For each user at each time t, they considered all open and available questions as possible candidates for users to select to answer, which was 328 questions per user per time. The random forest performed best, achieving a MRR= 0.446 meaning that selected questions are found on average in the 2nd or 3rd position. They further analyzed which features were most important and found that question features seem most informative, followed by user features. They also individual features by dropping them one by one and computing the accuracy of the random forest. Referal count of the question, question reputation and number of answers were ranked highest. Restricting the features to a subset containing the ones that were highest ranked they could improve the overall accuracy, indication that feature selection can increase performance and reduce computational effort at the same time.

**metrics**
* MAP@n
* Precision @ n

## IEEE 2015 - Exploiting User Feedback for Expert Finding in Community Question Answering <sub><sup><sub><sup>Needs better grammar proofreading</sup></sub></sup></sub>

The paper formalizes expert finding as a "learning to rank" problem, and presents a listwise learning approach called ListEF. The learned ranking function computes a score for every question-user pair, each pair a feature vector, and the outcome is evaluated by the corresponding answer score from the training data with a listwise loss function.

To identify user topical expertise, the authors first train a tagword topic model (TTM) to extract topics from questions, and then extract user expertise features by a COmpetition-based User exPertise Extraction (COUPE) method. At last, the ranking function is learned with user expertise features using LambdaMART.

TTM uses tag-word combinations and aggregates them at the corpus level, to overcome the data sparsity problem caused by short questions which makes alternatives like LDA perform poorly. The latent distributions on topics, tags and words are learned using Gibbs sampling. Later, topic distributions are inferred for the single documents. COUPE profiles every user by comparing their answer to a question to those of other users, and constructing counts of "wins", "ties" and "losses". Then, given a new question, the counts for previous questions are reshaped using topic similarity to the current question, and aggregated with simple descriptive statistics.

**metric** Given all people that answered a particular question, they just try to predict the ranking of those answers. They use NDCG@k as loss, which measures 'how much the predicted ranking has a different order'

## IEEE 2015 - Who Will Answer my Question on StackOverflow?


The paper aims to predict whether a certain user will answer a certain question, using a feature-based prediction approach and a social network based prediction approach.

The feature-based approach uses question and user features to train a Random Forest. To determine which tuples to feed the learning algorithm with, for every question it first computes a list of similar questions, and then only considers tuples of the original question with users who answered questions in the list. Beyond very simple features, question features include topic features obtained with LDA.

The social network based approach consists in building a social network graph with users, and predicting candidate answering users with a local search on the graph. The edges in the social network graph have weights corresponding to the amount of communication that took place between the users, with multipliers determined by the amount of tags in common and activity time in common.

The two approaches are combined by putting together the respective candidate users lists. This combined approach achieves 44% precision, 59% recall, and 49% F-measure (average across all test sets), but is not compared with any baselines.

## SAN 2013 - Routing Questions for Collaborative Answering in Community Question Answering

They also try to route questions to answerers, but in contrast to others, they route the question to "a group of answerers who would be willing to collaborate". Their motivation is that they investigated how the number of view of a thread is correlated with the number of answers, answer score and number of comments within the first hour, and how that all are statistically significant, so apperently "commenting in the early stage greatly improves the lasting value of a question".

To find the group of users most compatible to answer the question, they take into account four main features: Topic expertice of a user(exp), readiness of a user to answer a topic (cmt), availability of the user in the time (avail) and compatability of user1 to user2 (compat). They describe in detail how they compute these values, most importantly, in order to extract the topics, they propose taking cosine similarity between tags and do spectral clustering to obtain topics, and they show that it performs better than using LDA on the question text to extract topics.

The overall algorithm to find a GROUP of users goes as follows: They iteratively add users to the set, each time using the user with the maximum score when multiplying their feature scores (avail*exp*compat) for the group of answerers, and (avail*cmt*compat) fot predicting a group of commenters. They compare it to some baselines using MRR and so on.

## RecSys 2014 - Question Recommendation for Collaborative Question Answering Systems with RankSLDA

The author propose an algorithm called *RankSLDA* (Supervised LDA) to model the pairwise order of expertise of users with respect to a given question. This model simply extends the pre-existing *sLDA*, whose goal is to find the latent topics that best explain the observed responses, and does so by including in the LDA generative model the response variables y, which depend on the latent topic assignments z. The fitting is done with Gibbs sampling as usual, and then the obtained topic assignments z are used to train regression parameters for user topical expertise (to predict question answering scores) using logistic regression with l2 regularization.

The model is compared with baselines including LDA + Ranking (same as RankSLDA but without the supervised factor in Gibbs sampling) using basically all metrics such as P@N, nDCG@N (discounted cumulative gain), MAP, MRR. It's slightly better than LDA + Ranking, and much better than very simple baselines.


# Fairness
## ACM 2017 New Fairness Measures
same as NIPS 2017 Beyond Parity

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

* own idea: by adding the fairness penalty we also introduce information about group identity to the optimisation process. how does it compare if we give that information explicitly to the classifier. e.g. append to each user/item a 20 dimensional learned vector based on the group (i.e. for each group there is one learned vector). As the group belonging determines the rating (up to noise that is independent of the individual) this essentially reduces the problem to a 4x3 matrix. I am guessing doing this would result in best MSE and Fairness scores

## FA'IR: A Fair Top-k Ranking Algorithm
* affermative action type representation in a ranking algorithm. i.e. Frauenquote
* **input**
    * a set of individuals; each has a binary attribute whether it belongs to a protected group and a quality-score (e.g. GPA)
    * p in [0,1] the desired fraction of 'protected' people (like a quota)
    * significance level alpha
* **output**
    * a ranking where in each prefix of the ranking. (e.g. top 10, top 40, top 100) the fraction of protected people is not statistically significantly less then p. That is under H0 (in each place of ranking you throw a p-biased coin whether that person is protected or not) the probability to see a this unfair ranking is > alpha => we can't reject H0 and claim that it is unfair
    * they take care of multiple comparrisons ()
* **algorithm**
    * devide population into protected group and non protected group
    * sort in each group by quality-score => 2 queues
    * compute for each position how many protected individuals have to be until there at least to barely not jet reject H0
    * if you have to few protect people you add from that que, otherwise you add best from both queues




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
* cited in FA'IR
    * diversity: in top-k ranking you might also want to maximise the dissimilarity between items (not only fairness issue)


## Fairness in Algorithmic Decision-Making: Applications in Multi-Winner Voting, Machine Learning, and Recommender Systems

The paper provides an overview of fairness research in three domains: multi-winner voting, machine learning, and recommender systems. I just read the recommender system part. They also differ between fairness types, which is:

* Equalized Odds: Given the true outcome of a variable Y, and the group of a user A, the probability of prediction Y' is different - then unfair
* Equal opportunity: Given that Y=1 and same as above
* Statistical parity: Most simple case: Fair is that the probability for each class is the same for different groups
* Counterfactual Fairness: Same probabilities taking into account a context X
* Fairness through awareness: Algorithm is fair if it gives similar predictions for similar individuals (similarity metric)
* Individual Fairness: Define distance metric D, then D(predictions of individual A, predictions of individual B) should be smaller equal than D(individual A, individual B)


The also differ between different types of biases:
* Popularity bias: Favor of recommendation algorithms for popular items over items that may only be popular in small groups
* Observation bias: item displayed by the recommender system gets further reinforced in the choice by the agent over the period of time, leading to the increase in probability for the item to be retained in the system
* Systematic bias: biases that come from imbalance in the data are caused when a systematic bias is present in the data/ experience due to societal or historical features

Last, the research also differs between
* User& Group fairness
* Item fairness: item fairness should ensure that the various (past) areas of interest of a user need to be reflected with their corresponding proportions when making current recommendation
* Multiple stakeholder fairness: While traditional recommender systems focused specifically towards satisfaction of consumer by providing a set of relevant content, these multi-sided recommender systems face the problem of additionally optimizing preferences for providers as well as for platform. Fairness requires multiple parties to gain or lose equally with respect to the recommendations made.

The paper is a literature review talking about a lot of different approaches for each type. We might be able to get some inspiration when we try to reduce bias in the last part of the project, but for now it's hard to summarize.
