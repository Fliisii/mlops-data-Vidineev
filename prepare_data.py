from pathlib import Path
import argparse
import shutil

import pandas as pd
from sklearn.datasets import load_iris, load_wine

DATA_DIR = Path("data")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame = frame.sort_values(
        list(frame.columns), kind="mergesort"
    ).reset_index(drop=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["base", "wine", "iris_features"],
        help="Версия данных для создания",
    )
    mode = parser.parse_args().mode

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir()

    iris = load_iris(as_frame=True).frame.copy()

    if mode == "iris_features":
        iris["sepal_area"] = iris["sepal length (cm)"] * iris["sepal width (cm)"]
        iris["petal_area"] = iris["petal length (cm)"] * iris["petal width (cm)"]

    save_csv(iris, DATA_DIR / "iris.csv")

    if mode == "wine":
        wine = load_wine(as_frame=True).frame.copy()
        save_csv(wine, DATA_DIR / "wine.csv")

    print(f"Данные подготовлены: mode={mode}")
    for path in sorted(DATA_DIR.glob("*.csv")):
        print(f"{path}: {len(pd.read_csv(path))} строк")


if __name__ == "__main__":
    main()
