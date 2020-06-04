import pandas as pd
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


def get_vector(img_keypoints, p1, p2):
    """ Makes vector from joints with given numbers
        Делает верктора из точек по номерам

    Args:
        p1, p2: Номера пары точек, из которых состоит вектор

    Returns:
        вектор из р1 в р2
    """
    return (
        (img_keypoints[p2][0] - img_keypoints[p1][0]),
        (img_keypoints[p2][1] - img_keypoints[p1][0]),
    )


def cos_sim(a, b):
    """ Считает косинусное расстояние

    Args:
        а, b: вектора

    Returns:
        косинусное расстояние
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cos_sim_vertical(a):
    """ Считает косинусное расстояние c вертикалью

    Args:
        а: вектор

    Returns:
        косинусное расстояние c вертикалью
    """
    b = (0, 1)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def make_dataset(state):
    """ Создает датасет из сохраненного словаря с ключевыми точками

    Args:
        state: состояние (before или fall)

    Returns:
        prepare_data: подготовленный датасет
    """
    json_file_path = f"results/data_{state}.json"
    with open(json_file_path, "r") as j:
        contents = json.loads(j.read())

    prepare_data = prepare_dataset(contents)
    prepare_data = (
        pd.DataFrame.from_dict(prepare_data, orient="index")
        .reset_index()
        .rename(columns={"index": "name", 0: "poseKeypoints"})
    )

    prepare_data = prepare_data[["name", "poseKeypoints"]]

    prepare_data["Bedro_1"] = cos_sim_vertical_col(prepare_data, 8, 9)
    prepare_data["Bedro_2"] = cos_sim_vertical_col(prepare_data, 11, 12)
    prepare_data["Golen_1"] = cos_sim_vertical_col(prepare_data, 9, 10)
    prepare_data["Golen_2"] = cos_sim_vertical_col(prepare_data, 12, 13)
    prepare_data["Corpus"] = cos_sim_vertical_col(prepare_data, 1, 8)
    prepare_data["Taz"] = cos_sim_vertical_col(prepare_data, 9, 12)
    prepare_data["Telo"] = cos_sim_vertical_col(prepare_data, 1, 11)
    # prepare_data["Plechi"] = cos_sim_vertical_col(prepare_data, 2, 5)
    # prepare_data["Golova_1"] = cos_sim_vertical_col(prepare_data, 1, 17)
    # prepare_data["Golova_2"] = cos_sim_vertical_col(prepare_data, 1, 18)
    # prepare_data["Sgib_ruki_1"] = cos_sim_vertical_col(prepare_data, 5, 7)
    # prepare_data["Sgib_ruki_2"] = cos_sim_vertical_col(prepare_data, 2, 4)
    # prepare_data["Plecho_2"] = cos_sim_vertical_col(prepare_data, 2, 3)
    # prepare_data["Lokot_2"] = cos_sim_vertical_col(prepare_data, 3, 4)
    # prepare_data["Plecho_1"] = cos_sim_vertical_col(prepare_data, 5, 6)
    # prepare_data["Lokot_1"] = cos_sim_vertical_col(prepare_data, 6, 7)

    prepare_data = prepare_data.fillna(0)

    if state == "fall":
        prepare_data["Target"] = 1
    elif state == "before":
        prepare_data["Target"] = 0

    return prepare_data


def prepare_dataset(contents):
    """ Удаляет объекты без детекции ключевых точек

    Args:
        contents: словарь с ключевыми точками

    Returns:
        prepare_data: очищенный датасет
    """
    prepare_data = {}
    for key in contents:
        if type(contents[key]) == list:
            prepare_data[key] = contents[key]
    return prepare_data


def cos_sim_vertical_col(prepare_data, p1, p2):
    """ Добавляет признак - косинус между осями и вертикалью

    Args:
        prepare_data: подготовленный датасет
        p1: первая точка
        p2: вторая точка
    Returns:
        датасет с признаком
    """
    return prepare_data.poseKeypoints.apply(
        lambda x: cos_sim_vertical(get_vector(x, p1, p2))
    )


def get_all_dataset():
    """ Cоздает общий датасет с 1 и 0 классами

    Returns:
        df_all: общий датасет
    """

    # np.random.seed(17 + i * 5)

    df_before = make_dataset("before")
    df_fall = make_dataset("fall")
    df_all = df_before.append(df_fall).drop(columns=["name", "poseKeypoints"])

    # Перемешаем
    df_all = df_all.sample(frac=1).reset_index(drop=True)

    return df_all


def validate(df_all, part_for_val=3):
    """ Проверяет качество (precision) на отложенной выборке

    Returns:
        precision: точность предсказания
    """
    num_train = (part_for_val - 1) * (len(df_all) // part_for_val)
    df_train = df_all[:num_train]
    df_val = df_all[num_train:]

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(df_train.drop(columns="Target"), df_train["Target"])

    pred = clf.predict(df_val.drop(columns="Target"))

    precision = (df_val.Target.values == pred).astype(int).sum() / pred.shape[0]

    return precision


def train_and_export(df_all):
    clf = DecisionTreeClassifier(criterion="gini", random_state=0)
    clf.fit(df_all.drop(columns="Target"), df_all["Target"])
    dot = export_graphviz(
        clf,
        feature_names=df_all.drop(columns='Target').columns,
        out_file=None,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    dot = graphviz.Source(dot)
    dot.render('visualization','./results/', view=False)


if __name__ == "__main__":
    i = 0
    precision_mean = []
    while i < 10:
        df_all = get_all_dataset()
        precision = validate(df_all)
        precision_mean.append(precision)
        print("Точность на отложенной выборке: ", precision)
        i += 1
    print("Средняя точность на отложенных выборках: ", np.mean(precision_mean))
        
    train_and_export(df_all)