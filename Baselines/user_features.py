from data import Data
import numpy as np
from datetime import date
from functools import reduce
import pandas as pd

def to_taglist(string):
    if (not isinstance(string, str)) or len(string)==0:
        return []
    assert(string[0]=="<")
    assert(string[-1]==">")
    return string[1:-1].split("><")

def topic_reputation(data, tag_str, user_id, answer_creation):
    """
    computes the topic reputation for one user-question pair, as defined in section 3.3.1 in the paper.
    """
    tags = to_taglist(tag_str)
    query_string = "SELECT Id FROM Posts WHERE Tags "+" OR Tags ".join(["LIKE '%%<"+tags[i]+">%%'" for i in range(len(tags))])
    posts_with_tags = data.query(query_string)
    if len(posts_with_tags)==1: # dann kein richties tuple, sondern (123,) --> umwandeln in (123)
        posts_with_tags_ids = "("+str(posts_with_tags["id"].values.tolist()[0])+")"
    else:
        posts_with_tags_ids = tuple(posts_with_tags["id"].values.tolist())

    # asker_score = data.query("SELECT sum(Score) FROM Posts WHERE OwnerUserId={} AND PostTypeId=2 AND CreationDate<'{}' AND ParentId IN {}".format(asker_id, str(answer_creation), posts_with_tags_ids))
    user_score = data.query("SELECT sum(Score) as score FROM Posts WHERE OwnerUserId={} AND PostTypeId=2 AND CreationDate<'{}' AND ParentId IN {}".format(user_id, str(answer_creation), posts_with_tags_ids))
    # print(user_score.values[0])
    score = user_score.values[0][0]
    if score is not None:
        return score
    else:
        return 0

def topic_affinity(user_tags_str, question_tags_str):
    """
    Computes topic affinity for one user-question pair, as defined in section 3.3.1 in the paper
    In the paper, it is not clear what "user tags" refer to. Here, I simply took the tags of all
    questions posted by the user, as well as the tags of the question this user answered
    """
    user_tags = to_taglist(user_tags_str)
    unique_user_tags, counts = np.unique(user_tags, return_counts=True)
    user_pdf = counts/np.sum(counts)

    question_tags = to_taglist(question_tags_str)

    activated_tags = np.isin(unique_user_tags, question_tags)

    probs_of_activated = user_pdf[activated_tags]
    if len(probs_of_activated) > 0:
        prod = np.prod(probs_of_activated)
    else:
        prod = 0
    return prod


def get_user_features(data):
    ## USER FEATURE QUERIES
    # Number of answers posted by this user so far
    num_answers = data.query("SELECT OwnerUserId, count(OwnerUserId) as number_answers FROM Posts WHERE PostTypeId=2 AND OwnerUserId IS NOT NULL GROUP BY OwnerUserId")

    # Answering success:
    answering_success = data.query("SELECT OwnerUserId, count(OwnerUserId) as accepted_answers FROM Posts WHERE PostTypeId=2 AND OwnerUserId IS NOT NULL AND Id IN (SELECT AcceptedAnswerId FROM Posts WHERE AcceptedAnswerId IS NOT NULL) GROUP BY OwnerUserId")

    # Number of posts:
    num_questions = data.query("SELECT OwnerUserId, count(OwnerUserId) as num_questions FROM Posts WHERE PostTypeId=1 AND OwnerUserId IS NOT NULL GROUP BY OwnerUserId")

    # Number of questions:
    num_posts = data.query("SELECT OwnerUserId, count(OwnerUserId) as number_posts FROM Posts WHERE OwnerUserId IS NOT NULL GROUP BY OwnerUserId")

    ## For the next two, I took the sum of the scores, is there a different reputation field per post that I miss? Also, we could also take average score instead of sum
    # Question reputation:
    question_reputation = data.query("SELECT OwnerUserId, sum(Score) as question_reputation FROM Posts WHERE PostTypeId=1 AND OwnerUserId IS NOT NULL GROUP BY OwnerUserId")
    # Answer reputation:
    answer_reputation = data.query("SELECT OwnerUserId, sum(Score) as answer_reputation FROM Posts WHERE PostTypeId=2 AND OwnerUserId IS NOT NULL GROUP BY OwnerUserId")

    # Asking success: (Number of previous user questions that were identified as solved)
    prev_solved = data.query("SELECT OwnerUserId, count(OwnerUserId) as num_solved_questions FROM Posts INNER JOIN Users ON Posts.OwnerUserId=Users.Id WHERE PostTypeId=1 AND OwnerUserId IS NOT NULL AND AcceptedAnswerId IS NOT NULL GROUP BY OwnerUserId")

    # JOIN ALL RESULTS TO ONE DATAFRAME:
    joined_df = pd.merge(num_posts, num_questions, on="owneruserid", how="outer")
    for df in ["num_answers", "answering_success", "question_reputation", "answer_reputation", "prev_solved"]:
        joined_df = pd.merge(joined_df, eval(df), on="owneruserid", how="outer")
    reputation = data.query("SELECT Id as OwnerUserId, Reputation FROM Users")
    joined_df = pd.merge(joined_df, reputation, on="owneruserid", how="left").set_index("owneruserid")
    ## previous version: with merge
    # dfs = [reputation, num_posts, num_questions, num_answers, answering_success, question_reputation, answer_reputation, prev_solved]
    # joined_df = reduce(lambda left,right: pd.merge(left,right,on='owneruserid'), dfs).set_index("owneruserid")
    # print(joined_df.head(20))
    return joined_df

if __name__=="__main__":
    data = Data()

    data.set_time_range(start=date(year=2012, month=1, day=3), end=date(year=2014, month=5, day=1))

    joined_df = get_user_features(data)
    print(len(joined_df))
    print(joined_df.head(n=30))

    print("\nTest topic affinity function ...")
    print("Topic affinity of user 686 for question 2513:", topic_affinity(data, 919, 2513))

    print("\nTest topic reputation function ...")
    print("Topic reputation of user 686 for question 2513:", topic_reputation(data, 919, 726))
