from data import Data
import numpy as np
from datetime import date
from get_question_features import get_question_features
from user_features import get_user_features, topic_affinity, topic_reputation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Get Users with sufficient Answers:
min_answers = 10
# num_open_questions = 30

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
    dataframes=[]
    for index,row in answers_of_user.iterrows():
        question_answered = row["parentid"]
        answer_creation = row["creationdate"]
        answered_question = data.query("SELECT Id, body, ('{}'-CreationDate) AS questionage, OwnerUserId FROM Posts WHERE Id={}".format(str(answer_creation), question_answered)).set_index(["id"])
        if answered_question.empty or pd.isnull(answered_question).any().any(): # happens if answer is in specified time frame,but the corresponding question is not
            continue
        # select all questions (postid=1) which was created before that and has no accepted answer (PROBLEMATIC! TODO: check if accepted answer only afterwards)
        # Todo: add that user must not have answered the open question previously, and that user did not compose the question himself
        open_questions = data.query("SELECT Id, body, ('{}'-CreationDate) AS questionage, OwnerUserId FROM Posts WHERE PostTypeId=1 AND OwnerUserId IS NOT NULL AND CreationDate<'{}' AND (AcceptedAnswerId IS NULL) AND Id!={} ORDER BY questionage ASC".format(str(answer_creation), str(answer_creation),question_answered)).set_index(["id"])
        # label the answered question with 1, all others 0

        ## FIRST OPTION: randomly sample 20 out of ALL open questions
        # open_questions = open_questions.sample(min(20,len(open_questions)))
        ## SECOND OPTION: Take 30 most recent ones
        open_questions = open_questions.head(30)
        open_questions["label"] = 0
        answered_question["label"] = 1
        open_questions["decision_time"] = user_id*100 + time_user_counter
        answered_question["decision_time"] = user_id*100 + time_user_counter
        time_user_counter += 1
        dataframes.extend([open_questions, answered_question])
    question_df = pd.concat(dataframes) # concat all dataframes (all have the same columns)

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
    q_features["questionage"] = q_features["questionage"].dt.days +  (q_features["questionage"].dt.seconds)/(24*60*60) # convert questionage feature
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

    del q_features









#
