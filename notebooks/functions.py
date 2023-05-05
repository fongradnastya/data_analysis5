import matplotlib.pyplot as plt
import random

from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor


def build_response_graph(column):
    """
    Строит распределение классов в выборке
    :param column: массив рассматриваемых значений
    """
    plt.pie(column.value_counts(), labels=(0, 1))
    plt.title("Распределение значения response в выборке")
    plt.legend((0, 1))
    plt.show()


def get_best_score(best_scores, best_params, estimators=None):
    """
    Вычисляет информацию о лучшем скоре и параметрах модели
    :param best_scores: массив оценок работы моделей
    :param best_params: массив подобранных параметров модели
    :return: лучший скор
    """
    best_score = max(best_scores)
    best_id = best_scores.index(best_score)
    best_score = round(best_score, 3)
    best_param = best_params[best_id]
    if estimators:
        print("Best n_estimators:", best_param)
        print("Best estimator:", estimators[best_id])
    else:
        print("Best estimators combination:", best_param)
    print("Score:", best_score)
    return best_score


def test_models(estimators, parameters, x_train, y_train, x_test,
                y_test, is_bagg=True, is_regression=False):
    """
    Тестирует параметры бэггинга и бустинга
    :param estimators: модели для тестирования
    :param parameters: словарь тестируемых параметров
    :param x_train: тренировочная выборка
    :param y_train: ответы для тренировочной выборки
    :param x_test: тестовая выборка
    :param y_test: ответы для тестовой выборки
    :param is_bagg: тестируем ли работу баггинга
    :param is_regression: работаем ли над задачей регрессии
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
    build_parameters_graph(parameters["n_estimators"], train_acc, test_acc, estimators)
    return get_best_score(best_scores, best_params, estimators)


def build_parameters_graph(parameters, train_score, test_score, estimators,
                           is_regression=False):
    """
    Строит график зависимость score от параметров модели
    :param parameters: список параметров
    :param train_score: результаты на тренировочной выборке
    :param test_score: результаты на тестовой выборке
    :param estimators: использованные модели
    :param is_regression: рассматриваем ли задачу регрессии
    """
    fig, axs = plt.subplots(ncols=3, figsize=(20, 7))
    fig.subplots_adjust(hspace=0.4)
    name = str(estimators[0])
    plot_single_graph(parameters, name, axs[0], train_score[name],
                      test_score[name], regression=is_regression)
    name = str(estimators[1])
    plot_single_graph(parameters, name, axs[1], train_score[name],
                      test_score[name], regression=is_regression)
    name = str(estimators[2])
    plot_single_graph(parameters, name, axs[2], train_score[name],
                      test_score[name], regression=is_regression)


def plot_single_graph(parameters, estimator_name, ax, train_acc,
                      test_acc, regression=False):
    """
    Строит один из графиков зависимости точности от параметров модели
    :param parameters: список параметров
    :param estimator_name: название использованной модели
    :param ax: ось для построения графика
    :param train_acc: результаты на тренировочной выборке
    :param test_acc: результаты на тестовой выборке
    :param regression: рассматриваем ли задачу регрессии
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
    Строит столбчатую диаграмму score при стекинге
    :param comb_names: названия комбинаций моделей
    :param train_scores: результаты на тренировочной выборке
    :param test_scores: результаты на тестовой выборке
    :param regression: рассматриваем ли задачу регрессии
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
    Получает новую комбинацию моделей для стеккинга
    :param combinations: все возможные комбинации
    :param used: использованные комбинации
    :return: найденная комбинация, список использованных комбинаций
    """
    numb = random.randint(0, len(combinations) - 1)
    while numb in used:
        numb = random.randint(0, len(combinations) - 1)
    used.append(numb)
    combination = list(combinations[numb])
    return combination, used


def create_stacking_model(estimators, x_train, y_train, x_test, y_test,
                          sample_len=3, samples_cnt=5, is_regression=False):
    """
    Подбирает параметры для стеккинга
    :param estimators: набор моделей для исспользования
    :param x_train: тренировочная выборка
    :param y_train: ответы для тренировочной выборки
    :param x_test: тестовая выборка
    :param y_test: ответы для тестовой выборки
    :param sample_len: количество моделей в наборе
    :param samples_cnt: количество тестируемых наборов
    :param is_regression: рассматриваем ли задачу регрессии
    :return: скор лучшей модели
    """
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


def plot_bar_graph(ax, names, scoring, title):
    """
    Строит столбчатую диаграмму для score моделей
    :param ax: ось, на которой строим график
    :param names: значения параметров
    :param scoring: оценки при данных значениях параметров
    :param title: заголовок для графика
    :return:
    """
    ax.bar(names, scoring)
    ax.set_title(title)
    ax.tick_params("x", labelrotation=90)
    ax.set_ylim((0.5, 1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    for rect, score in zip(ax.patches, scoring):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                score, ha="center")
    ax.set_xlabel("Model", labelpad=15)
    ax.set_ylabel("Score", labelpad=15)


def compare_models(class_models, regression_models):
    """
    Строит график для сравнения скора моделей
    :param class_models: модели для классификации
    :param regression_models: модели для регрессии
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(25, 7))
    fig.subplots_adjust(hspace=0.4)
    plot_bar_graph(axs[0], regression_models["Model name"],
                   regression_models["R^2"], "Regression")
    plot_bar_graph(axs[1], class_models["Model name"],
                   class_models["Score"], "Classification")
