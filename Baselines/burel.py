import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from pipeline_utils import NamedColumnTransformer
import pandas as pd
import scipy.stats as st
import os

import features
from data import Data
from get_question_features import get_question_features
from user_features import get_user_features, topic_affinity, topic_reputation

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
NR_NEG_SAMPLES = 100 # TODO: change
NR_POS_SAMPLES = 200
ANS_PER_USER = 10

data = Data()
data.set_time_range(start=date(year=2012, month=1, day=3), end=date(year=2015, month=1, day=1))
# for data in data folder:
# data.set_time_range(start=date(year=2012, month=1, day=3), end=date(year=2015, month=1, day=1))

user_ids = data.query("SELECT OwnerUserId, c  FROM (SELECT OwnerUserId, count(OwnerUserId) as c FROM Posts WHERE PostTypeId=2 GROUP BY OwnerUserId) as tab WHERE c>%i"%min_answers)
user_ids = user_ids.sample(n=NR_POS_SAMPLES, random_state=50)
print("PROCESS ", len(user_ids), " USERS")
assert(np.all(user_ids["c"].values > min_answers))

user_ids = user_ids["owneruserid"].values
# do not take the users that are in the data directory already
file_list = os.listdir("data")
user_list_sofar = np.asarray([u.split(".")[0][4:] for u in file_list if u[0]=="u"]).astype(int)
user_ids = [u for u in user_ids if u not in user_list_sofar]
print("PROCESS ", len(user_ids), " USERS (not in data so far)")

## GET USER FEATURES
u_features = get_user_features(data) # need it for all data because we also need user features for the asker
# cannot simply get user tags for all users because then time component problematic

for j in range(len(user_ids)):
    time_user_counter = 0
    # take one user
    user_id = user_ids[j]
    # create list of his answered questions
    answers_of_user = data.query("SELECT Id, CreationDate, ParentId FROM Posts WHERE PostTypeID=2 AND OwnerUserId=%i"%user_id)
    answers_of_user = answers_of_user.sample(n=ANS_PER_USER, random_state=7)
    # get corresponding questions
    question_ids = tuple(answers_of_user["parentid"].values.tolist())
    answered_questions = data.query("SELECT Id, body, CreationDate as questionage, Tags as question_tags, OwnerUserId FROM Posts WHERE Id IN {}".format(question_ids))

    dataframes=[]
    for index,row in answers_of_user.iterrows():
        question_answered = row["parentid"]
        answer_creation = row["creationdate"]
        answered_question = (answered_questions[answered_questions.id==question_answered]).set_index(["id"])
        answered_question["questionage"] = answer_creation - answered_question["questionage"]

        if answered_question.empty or pd.isnull(answered_question).any().any(): # happens if answer is in specified time frame,but the corresponding question is not
            print("WARNING: ANSWERED QUESTION EMPTY (happens if answer is in specified time frame,but the corresponding question is not)")
            print("this answer is therfore left out")
            continue
        # select all questions (postid=1) which was created before that and has no accepted answer
        # where user has not answered the question (last part of where statement)
        open_questions = data.query("SELECT a.Id, body, ('{}'-CreationDate) AS questionage, Tags as question_tags, OwnerUserId FROM Posts a LEFT JOIN (SELECT Id, CreationDate as dateQ FROM Posts) b ON a.AcceptedAnswerId=b.Id WHERE PostTypeId=1 AND OwnerUserId IS NOT NULL AND CreationDate<'{}' AND (AcceptedAnswerId IS NULL OR b.dateQ>'{}') AND a.Id NOT IN {} ORDER BY questionage ASC".format(str(answer_creation), str(answer_creation), str(answer_creation), question_ids))
        open_questions["questionage"] = open_questions["questionage"].dt.days +  (open_questions["questionage"].dt.seconds)/(24*60*60) # convert questionage feature
        open_questions = sample_open_questions(open_questions, NR_NEG_SAMPLES).set_index(["id"])

        answered_question["questionage"] = answered_question["questionage"].dt.days +  (answered_question["questionage"].dt.seconds)/(24*60*60) # convert questionage feature
        open_questions["label"] = 0
        answered_question["label"] = 1

        ## GET USER TAGS (of his questions and answers before the time of this answer)
        tags_user_questions = data.query("SELECT Tags FROM Posts WHERE OwnerUserId={} AND PostTypeId=1 AND CreationDate<'{}'".format(user_id, str(answer_creation)))
        tags_user_answers = data.query("SELECT Tags FROM Posts WHERE Id IN (SELECT ParentId FROM Posts WHERE PostTypeId=2 AND OwnerUserId={} AND CreationDate<'{}')".format(user_id, str(answer_creation)))
        if len(tags_user_questions["tags"].values)>0 or len(tags_user_answers["tags"].values)>0:
            tag_list = tags_user_questions["tags"].values.tolist()
            tag_list.extend(tags_user_answers["tags"].values.tolist())
            tags = "".join(tag_list)
        else:
            tags = ""

        # concatenate and add information for both together
        answered_and_open = pd.concat([answered_question, open_questions])
        answered_and_open["user_tags"] = [tags for _ in range(len(answered_and_open))]
        answered_and_open["decision_time"] = user_id*100 + time_user_counter
        # answered_and_open["questionage"] = answered_and_open["questionage"].dt.days +  (answered_and_open["questionage"].dt.seconds)/(24*60*60) # convert questionage feature
        answered_and_open["user"] = [user_id for _ in range(len(answered_and_open))]

        ## Topic reputation
        answered_and_open["topic_reputation_user"] = answered_and_open.apply(lambda x: topic_reputation(data, x.question_tags, user_id, answer_creation), axis=1)
        answered_and_open["topic_reputation_asker"]  = answered_and_open.apply(lambda x: topic_reputation(data, x.question_tags, x.owneruserid, answer_creation), axis=1)

        time_user_counter += 1
        dataframes.append(answered_and_open)
        # dataframes.extend([open_questions, answered_question])
    question_df = pd.concat(dataframes) # concat all dataframes (all have the same columns)

    ### ADD FEATURES:
    # add normal question features
    q_features = get_question_features(question_df)

    # add features of the current user (the answerer) - same for all rows
    q_features = pd.merge(q_features, u_features, left_on = "user", right_index=True, how="left", sort=False)

    # add features of the answerer
    q_features = q_features.join(u_features, lsuffix='_user', rsuffix='_asker', on="owneruserid")

    ## Topic affinity
    q_features["topic_affinity_user"] = q_features.apply(lambda x: topic_affinity(x.user_tags, x.question_tags), axis=1)

    # fill nans with zero
    q_features = q_features.fillna(0)

    ## TESTING
    # rand_user = np.random.permutation(q_features.index)[0]
    # print(q_features.loc[rand_user,:])
    # print(np.asarray(q_features).shape)
    q_features = q_features.drop(["owneruserid", "body", "question_tags", "user_tags", "user"], axis=1)
    q_features.to_csv("data/user"+str(user_id)+".csv")
    print("SAVED USER", user_id)
    del q_features









#
