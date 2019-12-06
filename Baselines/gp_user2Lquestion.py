import pandas
import data
import data_utils
import utils
import pandas as pd
import gp_features
from datetime import timedelta
import numpy as np
import warnings
import pickle
import scipy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.preprocessing import normalize, StandardScaler


#Choose either "sklearn-GP" or "osgpr"
model_choice = "osgpr"
#For osgpr, M is the number of pseduo-points (for sparse approx)
M_points = 100

if model_choice == "osgpr":
    import tensorflow as tf
    import gpflow as GPflow
    import streaming_sparse_gp.osgpr as osgpr
    import streaming_sparse_gp.osgpr_utils as osgpr_utils

start_time_online_learning =  data_utils.make_datetime("01.01.2012 00:01")
hour_threshold_suggested_answer = 24
sigma = 1
beta = 0.4
n_preds = 5
time_delta_scores_after_posts = timedelta(days=2)

filter_nan_asker = True
filter_nan_answerer = True # i.e. skip events where asker or answerer field is empty. Also candidate questions with empty akser id will be ignored
only_open_questions_suggestable = False # if True candidate questions only contain questions which have no accepted answer at event time -> some people answer questions that already have an accepted answer

save_n_negative_suggestons = 1

pretraining_cache_file = "../cache/gp/pretraining.pickle"
redo_pretraining = False

all_events_file = "../cache/gp/all_events.pickle"
cached_data_file = "../cache/gp/cached_data.pickle"
redo_database_dumps = False

if redo_database_dumps:
    all_events_dataframe = data_utils.all_answer_events_dataframe(start_time=None, end_time=None, time_delta_scores_after_post=time_delta_scores_after_posts, filter_empty_asker=filter_nan_asker, filter_empty_target_user=filter_nan_answerer)
    all_events_dataframe.to_pickle(all_events_file)

    cached_data = data.DataHandleCached()
    with open(cached_data_file, "wb") as f:
        pickle.dump(cached_data, f)
else:
    all_events_dataframe = pd.read_pickle(all_events_file)

    with open(cached_data_file, "rb") as f:
        cached_data = pickle.load(f)


all_events_pretraining_dataframe = all_events_dataframe[all_events_dataframe.answer_date < start_time_online_learning]
all_events_main_timewindow = all_events_dataframe[all_events_dataframe.answer_date >= start_time_online_learning]


def is_user_answers_suggested_event(event):
    return event.question_age_at_answer <= timedelta(hours=hour_threshold_suggested_answer)

def get_suggestable_questions(time, filter_nan_asker_id=filter_nan_asker):

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

def compute_chance_success(n_candidates_list, n=n_preds):
    p = n/np.array(n_candidates_list)
    p[p>1] = 1 # if there n_preds smaller then candidate list
    return np.mean(p)

def print_intermediate_info(info_dict, current_time):
    if len(info_dict['event_time'])==0:
        print("empty info dict")
        return

    last_n = 10

    avg_candidates = np.mean(np.array(info_dict["n_candidates"])[-last_n:])
    most_recent_time = info_dict['event_time'][-1]

    percent_success = np.mean(np.array(info_dict['predicted_rank'][-last_n:]) != -1)

    chance_success = compute_chance_success(info_dict["n_candidates"][-last_n:])

    s = "{} | number of average candidates: {:.1f} | fraction_success : {:.3f} | chance_level success: {:.3f}".format(
        current_time, avg_candidates, percent_success, chance_success)
    print(s)




def top_N_ucb(mu, sigma, beta=beta, n=n_preds):
    upper_bounds = mu + sigma * np.sqrt(beta)
    # ids = utils.get_ids_of_N_largest(upper_bounds, n)
    sorted_ids = np.argsort(-upper_bounds)[:n]
    return sorted_ids # first is actually the one with the highest prediction


all_features_collection_raw = gp_features.GP_Feature_Collection(
    gp_features.GP_Features_affinity(),
    gp_features.GP_Features_TTM(),
    gp_features.GP_Features_Question(),
    gp_features.GP_Features_user())


def pretrain_gp_ucp(feature_collection, start_time, end_time):

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

        if not is_user_answers_suggested_event(event):
            feature_collection.update_pos_event(event)
        else:
            suggestable_questions = get_suggestable_questions(event.answer_date)
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

if redo_pretraining:
    st = data_utils.make_datetime("27.07.2010 17:06") #TODO wrong
    pretraining_result = pretrain_gp_ucp(all_features_collection_raw, start_time=st, end_time=start_time_online_learning)
    with open(pretraining_cache_file, "wb") as f:
        pickle.dump(pretraining_result, f)
else:
    with open(pretraining_cache_file, "rb") as f:
        pretraining_result = pickle.load(f)


all_features_collection, (training_set_for_gp, observed_labels) = pretraining_result
if model_choice == "osgpr":
    #Turn it into an array of 0 and 1s
    observed_labels = np.array([1.0 if i else 0.0 for i in observed_labels])[:, np.newaxis]
n_pretraining_samples = len(training_set_for_gp)
print("{} pretraining examples".format(n_pretraining_samples))
print(training_set_for_gp.shape)
# print(observed_labels.shape)


#With osgpr we pretrain immediately
if model_choice == "osgpr":
    persistent_scaler = StandardScaler()
    gp_input = persistent_scaler.fit_transform(training_set_for_gp)
    Z1 = gp_input[np.random.permutation(gp_input.shape[0])[0:M_points], :]
    #model = GPflow.sgpr.SGPR(gp_input, observed_labels, GPflow.kernels.RBF(gp_input.shape[1], ARD=True), Z=Z1)
    #model = GPflow.sgpr.SGPR(gp_input, observed_labels, GPflow.kernels.Linear(gp_input.shape[1], ARD=True), Z=Z1)
    model = GPflow.sgpr.SGPR(gp_input, observed_labels, osgpr_utils.CustLinearKernel(gp_input.shape[1], alpha=1e-5, ARD=True), Z=Z1)
    model.likelihood.variance = 0.001
    #model.kern.variance = 1.0
    model.kern.variance = np.ones(gp_input.shape[1])
    #model.kern.lengthscales = np.ones(gp_input.shape[1]) * 0.8
    model.optimize(disp=1, maxiter=50)
    #model.optimize(method=tf.train.AdamOptimizer(), maxiter=100)

info_dict = {'answer_id': list(), 'event_time': list(), 'user_id': list(), 'n_candidates': list(), 'predicted_rank': list()}

debug_all_questions_used_by_gp =list()
n_new_points = 0

for i, (_rowname, event) in enumerate(all_events_main_timewindow.iterrows()):

    assert(not np.isnan(event.answerer_user_id))
    assert(not np.isnan(event.asker_user_id))

    if not is_user_answers_suggested_event(event):
        # Don't just update the coupe, also add to the df as observation
        all_features_collection.update_pos_event(event)
        # TODO: add to what_algo_observed
    else:
        target_user_id = event.answerer_user_id
        actually_answered_id = event.question_id
        event_time = event.answer_date

        suggestable_questions = get_suggestable_questions(event.answer_date)
        if len(suggestable_questions) ==0:
            warnings.warn("For answer id {} (to question {}) there was not a single suggestable question".format(event.answer_id, event.question_id))
            continue


        assert(np.all(suggestable_questions.question_owner_user_id.notnull()))
        assert(np.any(suggestable_questions.question_id == actually_answered_id))

        # compute features
        features = all_features_collection.compute_features(target_user_id, suggestable_questions, event_time)
        # previous version: (I changed it because it is not necessary to give a list of target_user_id)
        # features = all_features_collection.compute_features(len(suggestable_questions)*[target_user_id], suggestable_questions, event_time)


        # # fit and predict with gaussian process
        if model_choice == "sklearn-GP":
            # print("starting GP")
            gp_input = StandardScaler().fit_transform(training_set_for_gp[-1000:])
            gpr = GaussianProcessRegressor(kernel=DotProduct(sigma_0=1.0), random_state=0, alpha=1e-5, normalize_y=False).fit(gp_input, observed_labels[-1000:])
            mu, sigma = gpr.predict(features, return_std=True)
        elif model_choice == "osgpr":
            #If we added new points, do an online update

            print("n_new_points right before condition: {}".format(n_new_points))
            if n_new_points > 0:
                new_gp_input = persistent_scaler.transform(training_set_for_gp[-n_new_points:])
                new_observed_labels = observed_labels[-n_new_points:]

                mu, Su, Zopt = osgpr_utils.get_mu_su(model)

                x_free = tf.placeholder('float64')
                model.kern.make_tf_array(x_free)
                X_tf = tf.placeholder('float64')
                with model.kern.tf_mode():
                    Kaa = tf.Session().run(
                        model.kern.K(X_tf),
                        feed_dict={x_free: model.kern.get_free_state(), X_tf: model.Z.value})

                Zinit = osgpr_utils.init_Z(Zopt, new_gp_input, use_old_Z=False)

                tf.reset_default_graph()
                #new_model = osgpr.OSGPR_VFE(new_gp_input, new_observed_labels, GPflow.kernels.RBF(new_gp_input.shape[1], ARD=True), mu, Su, Kaa, Zopt, Zinit)
                #new_model = osgpr.OSGPR_VFE(new_gp_input, new_observed_labels, GPflow.kernels.Linear(new_gp_input.shape[1], ARD=True), mu, Su, Kaa, Zopt, Zinit)
                new_model = osgpr.OSGPR_VFE(new_gp_input, new_observed_labels, osgpr_utils.CustLinearKernel(new_gp_input.shape[1], alpha=1e-5, ARD=True), mu, Su, Kaa, Zopt, Zinit)
                new_model.likelihood.variance = model.likelihood.variance.value
                new_model.kern.variance = model.kern.variance.value
                #new_model.kern.lengthscales = model.kern.lengthscales.value
                model = new_model
                model.optimize(disp=1, maxiter=30)
                #model.optimize(method=tf.train.AdamOptimizer(), maxiter=100)

            mu, var = model.predict_f(features)
            mu = np.squeeze(mu)
            sigma = np.squeeze(np.sqrt(var))
        else:
            raise NotImplementedError("This model hasn't been implemented yet")

        # print("mu", mu)
        # print("sigma", sigma)
        max_inds = top_N_ucb(mu, sigma) # this is the indexes of the predicted question that the user will answer
        # print("finished GP")
        # print("maximal indices", max_inds)

        rank_of_true_question = -1

        # update feature database
        for rank, selected_id in enumerate(max_inds):
            actually_suggested_question = suggestable_questions.iloc[selected_id]

            if actually_suggested_question.question_id == actually_answered_id:
                # the suggested question is what was actually answered
                all_features_collection.update_pos_event(event)
                rank_of_true_question = rank
            else:
                # this suggested question was not answered
                all_features_collection.update_neg_event(event) # i think all features so far ignore this


        # update training_data for gaaussian process

        suggested_questions_features = features.iloc[max_inds]
        suggested_questions_label = (suggestable_questions.iloc[max_inds].question_id.values == actually_answered_id)

        # what we want to save
        data_to_use_mask = utils.first_k_false_mask(suggested_questions_label, save_n_negative_suggestons) # take the true example + the save_n_negative_suggestions negative examples with the highest mean+sigma

        question_features_to_save = suggested_questions_features[data_to_use_mask]
        labels_to_save = suggested_questions_label[data_to_use_mask]
        if np.any(suggested_questions_label):
            assert(np.any(labels_to_save)) #if the true question was suggested we should better save that.


        # for debugging
        debug_all_questions_used_by_gp.append(suggestable_questions.iloc[max_inds][data_to_use_mask])


        assert(np.all(training_set_for_gp.columns == features.columns))

        n_new_points = question_features_to_save.shape[0]
        training_set_for_gp = pd.concat([training_set_for_gp, question_features_to_save])
        if model_choice == "osgpr":
            #Turn boolean into 0 and 1
            labels_to_save = np.array([1.0 if i else 0.0 for i in labels_to_save])[:, np.newaxis]
            observed_labels = np.concatenate((observed_labels, labels_to_save))
        else:
            observed_labels.extend(labels_to_save)

        assert(len(training_set_for_gp) == len(observed_labels))


        # update info
        info_dict["answer_id"].append(event.answer_id)
        info_dict["event_time"].append(event_time)
        info_dict["user_id"].append(target_user_id)
        info_dict["n_candidates"].append(len(suggestable_questions))
        info_dict["predicted_rank"].append(rank_of_true_question)

        # print('pred rank', info_dict['predicted_rank'])


        if i%1==0:
            print('mu', mu)
            print('sigma', sigma)
            print('label', suggested_questions_label)
            print_intermediate_info(info_dict, event.answer_date)

    if i % 1000 == 0 and i>100:
        if len(debug_all_questions_used_by_gp) != 0:
            debug_used_questions=pd.concat(debug_all_questions_used_by_gp, axis=0)
            assert(len(debug_used_questions) == len(training_set_for_gp[n_pretraining_samples:]))
            debug_used_questions.loc[:, "label"] = observed_labels[n_pretraining_samples:]
            debug_used_questions.to_csv("events_used_by_gp.csv")
            training_set_for_gp.to_csv("training_set_for_gp.csv")
            gp_info_dict = pd.DataFrame(data = info_dict)
            gp_info_dict.to_csv("gp_run_info_dict_{}.csv".format(model_choice))
