import matplotlib.pyplot as plt
import random
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor


def get_best_score(best_scores, best_params, estimators=None):
    """

    :param best_scores:
    :param best_params:
    :return:
    """
    best_score = max(best_scores)
    best_id = best_scores.index(best_score)
    best_score = round(best_score, 3)
    best_param = best_params[best_id]
    print("Best parameter:", best_param)
    if estimators:
        print("Best estimator:", estimators[best_id])
    print("Score:", best_score)
    return best_score


def test_models(estimators, parameters, x_train, y_train, x_test,
                y_test, is_bagg=True, is_regression=False):
    """

    :param estimators:
    :param parameters:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param is_bagg:
    :param is_regression:
    :return:
    """
    train_acc = {}
    test_acc = {}
    best_scores = []
    best_params = []
    for estimator in estimators:
        train_acc[str(estimator)] = []
        test_acc[str(estimator)] = []
        for parameter in parameters["n_estimators"]:
            if not is_regression:
                if is_bagg:
                    model = BaggingClassifier(estimator,
                                              n_estimators=parameter,
                                              random_state=42)
                else:
                    model = AdaBoostClassifier(estimator,
                                               n_estimators=parameter,
                                               random_state=42)
                model.fit(x_train, y_train)
                y_pred_train = model.predict_proba(x_train)[:, 1]
                y_pred_test = model.predict_proba(x_test)[:, 1]
                train_acc[str(estimator)].append(
                    roc_auc_score(y_train, y_pred_train))
                test_acc[str(estimator)].append(
                    roc_auc_score(y_test, y_pred_test))
            else:
                if is_bagg:
                    model = BaggingRegressor(estimator,
                                             n_estimators=parameter,
                                             random_state=42)
                else:
                    model = AdaBoostRegressor(estimator,
                                              n_estimators=parameter,
                                              random_state=42)
                model.fit(x_train, y_train)
                train_acc[str(estimator)].append(model.score(x_train, y_train))
                test_acc[str(estimator)].append(model.score(x_test, y_test))
        best_scores.append(max(test_acc[str(estimator)]))
        best_id = test_acc[str(estimator)].index(best_scores[-1])
        best_params.append(parameters["n_estimators"][best_id])
    build_graph(parameters["n_estimators"], train_acc, test_acc, estimators)
    return get_best_score(best_scores, best_params, estimators)


def build_graph(parameters, train_score, test_score, estimators,
                is_regression=False):
    """

    :param parameters:
    :param train_score:
    :param test_score:
    :param estimators:
    :param is_regression:
    :return:
    """
    fig, axs = plt.subplots(ncols=3, figsize=(20, 7))
    fig.subplots_adjust(hspace=0.4)
    name = str(estimators[0])
    plot_accuracy_graph(parameters, name, axs[0], train_score[name],
                        test_score[name], regression=is_regression)
    name = str(estimators[1])
    plot_accuracy_graph(parameters, name, axs[1], train_score[name],
                        test_score[name], regression=is_regression)
    name = str(estimators[2])
    plot_accuracy_graph(parameters, name, axs[2], train_score[name],
                        test_score[name], regression=is_regression)


def plot_accuracy_graph(parameters, estimator_name, ax, train_acc,
                        test_acc, regression=False):
    """
    Строит график зависимости точности от параметров модели
    :param parameters: список параметров
    :param parameter_name: название исследуемого параметра
    :param train_acc: производительность на тренировочной выборке
    :param test_acc: производительность на тестовой выборке
    :param cross_acc: производительность при перекрёстной проверке
    :param regression: используем ли модель регрессии
    """
    ax.plot(parameters, train_acc, label="Train")
    ax.plot(parameters, test_acc, label="Test")
    ax.set_xlabel("n_estimators", labelpad=15)
    ax.set_title(estimator_name)
    if not regression:
        ax.set_ylabel("ROC-AUC", labelpad=15)
    else:
        ax.set_ylabel("R^2", labelpad=15)
    ax.legend()


def build_stacking_graph(comb_names, train_scores, test_scores,
                         regression=False):
    """

    :param comb_names:
    :param train_scores:
    :param test_scores:
    :param regression:
    :return:
    """
    fig, ax = plt.subplots()
    fig.suptitle("Stacking")
    ax.bar(comb_names, train_scores, color="b", label="train_score")
    ax.bar(comb_names, test_scores, color="r", label="test_score")
    ax.set_xticks(range(len(comb_names)))
    ax.set_xticklabels(comb_names)
    ax.set_xlabel("model combinations")
    ax.set_ylim([min(test_scores) - 0.01, max(train_scores) + 0.01])
    ax.tick_params("x", labelrotation=90)
    if not regression:
        ax.set_ylabel("ROC-AUC", labelpad=15)
    else:
        ax.set_ylabel("R^2", labelpad=15)
    ax.legend()
    plt.tight_layout()
    plt.show()


def get_new_combinations(combinations, used):
    """

    :param combinations:
    :param used:
    :return:
    """
    numb = random.randint(0, len(combinations) - 1)
    while numb in used:
        numb = random.randint(0, len(combinations) - 1)
    used.append(numb)
    combination = list(combinations[numb])
    return combination, used


def create_stacking_model(estimators, x_train, y_train, x_test, y_test,
                          sample_len=3, samples_cnt=5, is_regression=False):
    new_combinations = list(combinations(estimators, sample_len))
    names = []
    train = []
    test = []
    used_combinations = []
    for i in range(samples_cnt):
        combination, used_combinations = get_new_combinations(
            new_combinations, used_combinations)
        if not is_regression:
            model = StackingClassifier(combination)
            model.fit(x_train, y_train)
            y_pred_train = model.predict_proba(x_train)[:, 1]
            y_pred_test = model.predict_proba(x_test)[:, 1]
            train.append(roc_auc_score(y_train, y_pred_train))
            test.append(roc_auc_score(y_test, y_pred_test))
        else:
            model = StackingRegressor(combination,
                                      final_estimator=RandomForestRegressor())
            model.fit(x_train, y_train)
            train.append(model.score(x_train, y_train))
            test.append(model.score(x_test, y_test))
        name = ""
        for item, _ in combination:
            name += item + " "
        names.append(name)
    build_stacking_graph(names, train, test, regression=is_regression)
    return get_best_score(test, names)
