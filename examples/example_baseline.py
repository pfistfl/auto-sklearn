"""
=======
Baseline
=======
"""

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification



# Scenario 1: Search through initial 5 meta-learning configurations.
# We set a runcount_limit of 5 for smac (5 iters) and 
# set the ensemble size to 1 (we only want one model).
# To stop the ensembling process, ensemble_nbest is set to 5 and used
# to terminate the search.
def main():
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=30,
        ensemble_nbest=5,
        ensemble_size=1,
        initial_configurations_via_metalearning=5,
        include_estimators=['random_forest'],
        include_preprocessors=['no_preprocessing'],
        tmp_folder='/home/flo/autosklearn_holdout_example_tmp',
        smac_scenario_args={'runcount_limit': 5},
        delete_tmp_folder_after_terminate=False,
        disable_evaluator_output=False,
        # 'holdout' with 'train_size'=0.67 is the default argument setting
        # for AutoSklearnClassifier. It is explicitly specified in this example
        # for demonstrational purpose.
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 3}
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')
    automl.refit(X_train, y_train)

    # Print the final ensemble constructed by auto-sklearn.
    print(automl.show_models())
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
