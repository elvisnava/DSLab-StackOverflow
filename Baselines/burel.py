from data import Data
import numpy as np
from datetime import date
from get_question_features import get_question_features
from user_features import get_user_features, topic_affinity, topic_reputation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import scipy.stats as st

def sample_open_questions(open_questions, sample_size):
    ## DISTRIBUTION APPROACH:
    age_vals = open_questions["questionage"].values
    # rans = sorted(st.expon.rvs(loc=0, scale=3, size=sample_size))
    rans = sorted(st.gilbrat.rvs(-0.3530395997092245, 1.3032193696909253, size=sample_size))
    final_inds = []
    for i in range(len(age_vals)):
        if len(rans)==0:
            break
        if age_vals[i]>rans[0]:
            final_inds.append(i)
            del rans[0]
            # print(rans)
    open_questions = open_questions.loc[final_inds]
    ## FIRST OPTION: randomly sample 20 out of ALL open questions
    # open_questions = open_questions.sample(min(20,len(open_questions)))
    ## SECOND OPTION: Take 30 most recent ones
    # open_questions = open_questions.head(sample_size)

    return open_questions

## HYPER PARAMETERS
# Get Users with sufficient Answers:
min_answers = 10
NR_NEG_SAMPLES = 100

data = Data()
data.set_time_range(start=date(year=2014, month=1, day=3), end=date(year=2014, month=5, day=1))

user_ids = data.query("SELECT OwnerUserId, c  FROM (SELECT OwnerUserId, count(OwnerUserId) as c FROM Posts WHERE PostTypeId=2 GROUP BY OwnerUserId) as tab WHERE c>%i"%min_answers)
assert(np.all(user_ids["c"].values > min_answers))

user_ids = user_ids["owneruserid"].values

## GET USER AND FEATURES
u_features = get_user_features(data)
# u_features = u_features.loc[u_features['owneruserid'].isin(user_ids)] # not needed because we also need the users who have asked the question, they might have less than 5 posts


for j in range(40):
    time_user_counter = 0
    # take one user
    user_id = user_ids[j]
    # create list of his answered questions
    answers_of_user = data.query("SELECT Id, CreationDate, ParentId FROM Posts WHERE PostTypeID=2 AND OwnerUserId=%i"%user_id)
    # get corresponding questions
    question_ids = tuple(answers_of_user["parentid"].values.tolist())
    answered_questions = data.query("SELECT Id, body, CreationDate as questionage, OwnerUserId FROM Posts WHERE Id IN {}".format(question_ids))

    dataframes=[]
    for index,row in answers_of_user.iterrows():
        question_answered = row["parentid"]
        answer_creation = row["creationdate"]
        answered_question = (answered_questions[answered_questions.id==question_answered]).set_index(["id"])
        answered_question["questionage"] = answer_creation - answered_question["questionage"]
        # answered_question = answered_question.drop(["creationdate"], axis=1)
        # data.query("SELECT Id, body, ('{}'-CreationDate) AS questionage, OwnerUserId FROM Posts WHERE Id={}".format(str(answer_creation), question_answered)).set_index(["id"])
        if answered_question.empty or pd.isnull(answered_question).any().any(): # happens if answer is in specified time frame,but the corresponding question is not
            print("WARNING: ANSWERED QUESTION EMPTY (happens if answer is in specified time frame,but the corresponding question is not)")
            print("this answer is therfore left out")
            continue
        # select all questions (postid=1) which was created before that and has no accepted answer (PROBLEMATIC! TODO: check if accepted answer only afterwards)
        # where user has not answered the question (last part of where statement)
        open_questions = data.query("SELECT a.Id, body, ('{}'-CreationDate) AS questionage, OwnerUserId FROM Posts a LEFT JOIN (SELECT Id, CreationDate as dateQ FROM Posts) b ON a.AcceptedAnswerId=b.Id WHERE PostTypeId=1 AND OwnerUserId IS NOT NULL AND CreationDate<'{}' AND (AcceptedAnswerId IS NULL OR b.dateQ>'{}') AND a.Id NOT IN {} ORDER BY questionage ASC".format(str(answer_creation), str(answer_creation), str(answer_creation), question_ids))
        # open_questions = data.query("SELECT Id, body, ('{}'-CreationDate) AS questionage, OwnerUserId FROM Posts WHERE PostTypeId=1 AND OwnerUserId IS NOT NULL AND CreationDate<'{}' AND (AcceptedAnswerId IS NULL) AND Id NOT IN {} ORDER BY questionage ASC".format(str(answer_creation), str(answer_creation), question_ids))

        open_questions["questionage"] = open_questions["questionage"].dt.days +  (open_questions["questionage"].dt.seconds)/(24*60*60) # convert questionage feature
        open_questions = sample_open_questions(open_questions, NR_NEG_SAMPLES).set_index(["id"])

        answered_question["questionage"] = answered_question["questionage"].dt.days +  (answered_question["questionage"].dt.seconds)/(24*60*60) # convert questionage feature
        open_questions["label"] = 0
        answered_question["label"] = 1
        open_questions["decision_time"] = user_id*100 + time_user_counter
        answered_question["decision_time"] = user_id*100 + time_user_counter
        time_user_counter += 1
        dataframes.extend([open_questions, answered_question])
    question_df = pd.concat(dataframes) # concat all dataframes (all have the same columns)
    # print(question_df.head(32))

    ### ADD FEATURES:
    # add normal question features
    q_features = get_question_features(question_df)

    # add features of the answerer
    q_features = q_features.join(u_features, lsuffix='_caller', rsuffix='_other', on="owneruserid")

    # add features of the current user (the answerer) - same for all rows
    user_feature_row = u_features.loc[user_id,:]
    for j, col in enumerate(u_features.columns):
        val = user_feature_row[col]
        q_features[col+"_u"] = [val for _ in range(len(q_features))]
    # add other special features
    # q_features["questionage"] = q_features["questionage"].dt.days +  (q_features["questionage"].dt.seconds)/(24*60*60) # convert questionage feature
    q_features["question_id"] = q_features.index.map(lambda x: x)
    q_features["topic_affinity_user"] = q_features["question_id"].apply(lambda x: topic_affinity(data, user_id, x))
    q_features["topic_affinity_asker"] = q_features.apply(lambda x: topic_affinity(data, x.owneruserid, x.question_id), axis=1)
    q_features["topic_reputation_user"] = q_features["question_id"].apply(lambda x: topic_reputation(data, user_id, x))
    q_features["topic_reputation_asker"] = q_features.apply(lambda x: topic_reputation(data, x.owneruserid, x.question_id), axis=1)

    # fill nans with zero
    q_features = q_features.fillna(0)

    ## TESTING
    # rand_user = np.random.permutation(q_features.index)[0]
    # print(q_features.loc[rand_user,:])
    # print(np.asarray(q_features).shape)
    q_features = q_features.drop(["owneruserid", "body", "question_id"], axis=1)
    q_features.to_csv("data/user"+str(user_id)+".csv")
    print("SAVED USER", user_id)
    del q_features









#
