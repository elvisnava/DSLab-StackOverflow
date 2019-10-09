from data import Data
import numpy as np
from datetime import date
from functools import reduce
import pandas as pd


def topic_reputation(data, user, question_id):
    """
    computes the topic reputation for one user-question pair, as defined in section 3.3.1 in the paper.
    """
    tags = data.query("SELECT Tags FROM Posts WHERE Id=%i"%question_id)
    assert(len(tags["tags"].values)==1) # just a string is returned, one post id --> only one value
    tag_list = (tags["tags"].values)[0]
    tag_list =  [tag+">" for tag in tag_list.split(">") if tag]
    # print(tag_list)
    score_sum = 0
    for t in tag_list:
        # sum up scores for the answers (postTypeId=2) of user user where the cooresponding question (the parent) had the current tag
        scores = data.query("SELECT sum(Score) FROM Posts WHERE OwnerUserId={} AND PostTypeId=2 AND ParentId IN (SELECT Id FROM Posts WHERE Tags LIKE '{}')".format(user,t))
        summed_score = scores.values[0][0]
        if summed_score is not None:
            score_sum+=summed_score # sum up for all tags in tag_list
    return score_sum

def topic_affinity(data, user, question_id):
    """
    Computes topic affinity for one user-question pair, as defined in section 3.3.1 in the paper
    In the paper, it is not clear what "user tags" refer to. Here, I simply took the tags of all
    questions posted by the user, as well as the tags of the question this user answered
    """
    # tags for questions and answers of this user:
    tags_user_questions = data.query("SELECT Tags FROM Posts WHERE OwnerUserId=%i AND PostTypeId=1"%user)
    tags_user_answers = data.query("SELECT Tags FROM Posts WHERE Id IN (SELECT ParentId FROM Posts WHERE PostTypeId=2 AND OwnerUserId=%i)"%user)
    # tags for this particular question:
    tags_q = data.query("SELECT Tags FROM Posts where Id=%i"%question_id)
    assert(len(tags_q["tags"])==1)
    tags_question = [tag+">" for tag in tags_q["tags"][0].split(">") if tag]
    # put all question and user tags in one list:
    tag_list = []
    for t in list(tags_user_questions["tags"].values):
        if t is not None:
            post_tags = [tag+">" for tag in t.split(">") if tag]
            tag_list.extend(post_tags)
    for t in list(tags_user_answers["tags"].values):
        if t is not None:
            post_tags = [tag+">" for tag in t.split(">") if tag]
            tag_list.extend(post_tags)
    # transform to probabilities:
    unique_tags, tag_counts = np.unique(tag_list, return_counts=True)
    tag_probs = tag_counts/sum(tag_counts)
    prob_list = []
    for t_q in tags_question:
        if t_q in unique_tags:
            prob_list.append(tag_probs[unique_tags.tolist().index(t_q)])
    # return likelihood:
    if len(prob_list)==0:
        return 0 # if no intersection of user and question tags
    else:
        return np.prod(prob_list) # product of probabilities --> likelihood


def get_user_features(data):
    ## USER FEATURE QUERIES
    # Number of answers posted by this user so far
    num_answers = data.query("SELECT OwnerUserId, count(OwnerUserId) as number_answers FROM Posts WHERE PostTypeId=2 GROUP BY OwnerUserId")

    # Get reputation
    reputation = data.query("SELECT Id as OwnerUserId, Reputation FROM Users")
    # reputation2 = data.query("SELECT OwnerUserId, sum(Score) FROM Posts GROUP BY OwnerUserId") # reputation as sum of scores, totally different result than reputation
    # joined_df = reputation2.set_index('owneruserid').join(reputation.set_index('id')) # comment in for combined table

    # Answering success:
    answering_success = data.query("SELECT OwnerUserId, count(OwnerUserId) as accepted_answers FROM Posts WHERE PostTypeId=2 AND Id IN (SELECT AcceptedAnswerId FROM Posts WHERE AcceptedAnswerId IS NOT NULL) GROUP BY OwnerUserId")
    # joined_df = num_answers.set_index('owneruserid').join(answering_success.set_index('owneruserid')) # comment in for combined table

    # Number of posts:
    num_questions = data.query("SELECT OwnerUserId, count(OwnerUserId) as num_questions FROM Posts WHERE PostTypeId=1 GROUP BY OwnerUserId")

    # Number of questions:
    num_posts = data.query("SELECT OwnerUserId, count(OwnerUserId) as number_posts FROM Posts GROUP BY OwnerUserId")

    ## For the next two, I took the sum of the scores, is there a different reputation field per post that I miss? Also, we could also take average score instead of sum
    # Question reputation:
    question_reputation = data.query("SELECT OwnerUserId, sum(Score) as accumulated_score FROM Posts WHERE PostTypeId=1 GROUP BY OwnerUserId")
    # Answer reputation:
    answer_reputation = data.query("SELECT OwnerUserId, sum(Score) as accumulated_score FROM Posts WHERE PostTypeId=2 GROUP BY OwnerUserId")

    # Asking success: (Number of previous user questions that were identified as solved)
    prev_solved = data.query("SELECT OwnerUserId, count(OwnerUserId) as num_solved_questions FROM Posts INNER JOIN Users ON Posts.OwnerUserId=Users.Id WHERE PostTypeId=1 AND AcceptedAnswerId IS NOT NULL GROUP BY OwnerUserId")
    # joined_df = joined_df.join(prev_solved.set_index('owneruserid')) # comment in for combined table


    # JOIN ALL RESULTS TO ONE DATAFRAME:
    dfs = [reputation, num_posts, num_questions, num_answers, answering_success, question_reputation, answer_reputation, prev_solved]
    joined_df = reduce(lambda left,right: pd.merge(left,right,on='owneruserid'), dfs)

    return joined_df

if __name__=="__main__":
    data = Data()

    data.set_time_range(start=date(year=2014, month=1, day=3), end=date(year=2014, month=5, day=1))

    joined_df = get_user_features(data)
    print(joined_df.head(n=30))

    print("\nTest topic affinity function ...")
    print("Topic affinity of user 686 for question 2513:", topic_affinity(data, 686, 2513))

    print("\nTest topic reputation function ...")
    print("Topic reputation of user 686 for question 2513:", topic_reputation(data, 686, 726))
