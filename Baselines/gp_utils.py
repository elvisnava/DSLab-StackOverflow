import numpy as np
import pandas as pd
from datetime import timedelta

def is_user_answers_suggested_event(event, hour_threshold_suggested_answer):
    return event.question_age_at_answer <= timedelta(hours=hour_threshold_suggested_answer)

def get_suggestable_questions(time, cached_data, only_open_questions_suggestable, hour_threshold_suggested_answer, filter_nan_asker_id):

    open_questions = cached_data.existing_questions_at_time(time, only_open_questions=only_open_questions_suggestable)
    mask_young_enough = (open_questions.question_date >= time - timedelta(hours=hour_threshold_suggested_answer))

    if filter_nan_asker_id:
        mask_known_asker = open_questions.question_owner_user_id.notnull()
        mask = mask_young_enough & mask_known_asker
    else:
        mask = mask_young_enough

    return open_questions[mask]

def optimising_dummy_func(obj_func, initial_theta, bounds):
    func_val = obj_func(initial_theta)
    return initial_theta, func_val


def argmax_ucb(mu, sigma, beta):
    return np.argmax(mu + sigma * np.sqrt(beta))

def mrr_gp(ranks):
    inv = 1/ranks
    inv[inv<0] = 0
    return np.mean(inv)

def compute_chance_success(n_candidates_list, n):
    p = n/np.array(n_candidates_list)
    p[p>1] = 1 # if there n_preds smaller then candidate list
    return np.mean(p)

def print_intermediate_info(info_dict, current_time, n_preds):
    if len(info_dict['event_time'])==0:
        print("empty info dict")
        return

    last_n = 10

    avg_candidates = np.mean(np.array(info_dict["n_candidates"])[-last_n:])
    most_recent_time = info_dict['event_time'][-1]

    percent_success = np.mean(np.array(info_dict['predicted_rank'][-last_n:]) != -1)

    chance_success = compute_chance_success(info_dict["n_candidates"][-last_n:], n_preds)

    s = "{} | number of average candidates: {:.1f} | fraction_success : {:.3f} | chance_level success: {:.3f}".format(
        current_time, avg_candidates, percent_success, chance_success)
    print(s)




def top_N_ucb(mu, sigma, beta, n):
    upper_bounds = mu + sigma * np.sqrt(beta)
    # ids = utils.get_ids_of_N_largest(upper_bounds, n)
    sorted_ids = np.argsort(-upper_bounds)[:n]
    return sorted_ids # first is actually the one with the highest prediction

def pretrain_gp_ucp(feature_collection, all_events_pretraining_dataframe, hour_threshold_suggested_answer, cached_data, only_open_questions_suggestable,
                    filter_nan_asker_id, start_time, end_time):

    all_feates_collector = list()
    all_label_collector = list() # list of 1d numpy arrays


    n_candidates_collector = list()

    for i, (_rowname, event) in enumerate(all_events_pretraining_dataframe.iterrows()):
        assert(not np.isnan(event.answerer_user_id))
        assert(not np.isnan(event.asker_user_id))

        if i%100 ==0 :
            avg_candidates = np.mean(n_candidates_collector)
            print("Preptraining at {}| on average {} candidates in the last {} suggested_question_events".format(event.answer_date, avg_candidates, len(n_candidates_collector)))
            n_candidates_collector = list()

        if not is_user_answers_suggested_event(event, hour_threshold_suggested_answer):
            feature_collection.update_pos_event(event)
        else:
            suggestable_questions = get_suggestable_questions(event.answer_date, cached_data, only_open_questions_suggestable, hour_threshold_suggested_answer, filter_nan_asker_id)
            if len(suggestable_questions) ==0:
                warnings.warn("For answer id {} (to question {}) there was not a single suggestable question".format(event.answer_id, event.question_id))
                continue

            n_candidates_collector.append(len(suggestable_questions))

            feats = feature_collection.compute_features(event.answerer_user_id, suggestable_questions, event.answer_date)
            label = suggestable_questions.question_id.values == event.question_id


            assert(np.any(label))
            assert(np.all(suggestable_questions.question_owner_user_id.notnull()))


            all_feates_collector.append(feats)
            all_label_collector.append(label)

            # TODO I don't update the negative event here

            feature_collection.update_pos_event(event)

    all_feats = pd.concat(all_feates_collector, axis=0)
    all_label = np.concatenate(all_label_collector, axis=0).tolist()

    return feature_collection, (all_feats, all_label)
