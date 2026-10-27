from pathlib import Path
import argparse
import hashlib

import pandas as pd
from sklearn.datasets import load_iris, load_wine

DATA_DIR = Path("data")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_frame(name: str, mode: str) -> pd.DataFrame:
    if name == "iris":
        frame = load_iris(as_frame=True).frame.copy()
        if mode == "iris_features":
            frame["sepal_area"] = (
                frame["sepal length (cm)"] * frame["sepal width (cm)"]
            )
            frame["petal_area"] = (
                frame["petal length (cm)"] * frame["petal width (cm)"]
            )
        return frame

    return load_wine(as_frame=True).frame.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["base", "wine", "iris_features"])
    mode = parser.parse_args().mode

    expected_files = {"iris.csv"}
    if mode == "wine":
        expected_files.add("wine.csv")

    actual_files = {p.name for p in DATA_DIR.glob("*.csv")} if DATA_DIR.exists() else set()
    if actual_files != expected_files:
        raise SystemExit(
            f"ОШИБКА: ожидались файлы {sorted(expected_files)}, "
            f"найдены {sorted(actual_files)}"
        )

    for name in sorted(expected_files):
        path = DATA_DIR / name
        actual = pd.read_csv(path)
        base_name = path.stem
        expected = expected_frame(base_name, mode)

        expected = expected.sort_values(
            list(expected.columns), kind="mergesort"
        ).reset_index(drop=True)

        if any(str(col).startswith("Unnamed:") for col in actual.columns):
            raise SystemExit(f"ОШИБКА: обнаружен лишний индекс в {path}")

        if len(actual) != len(expected):
            raise SystemExit(
                f"ОШИБКА: {path}: строк {len(actual)}, ожидалось {len(expected)}"
            )

        if list(actual.columns) != list(expected.columns):
            raise SystemExit(
                f"ОШИБКА: неверные столбцы в {path}\n"
                f"Факт: {list(actual.columns)}\n"
                f"Ожидалось: {list(expected.columns)}"
            )

        pd.testing.assert_frame_equal(
            actual, expected, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9
        )

        print(f"\nФайл: {path}")
        print(f"SHA-256: {sha256(path)}")
        print(f"Размер: {actual.shape[0]} строк × {actual.shape[1]} столбцов")
        print(f"Столбцы: {list(actual.columns)}")
        print(f"Пропуски: {int(actual.isna().sum().sum())}")
        print("Проверка содержимого: OK")

    print(f"\nВсе проверки пройдены: mode={mode}")


if __name__ == "__main__":
    main()
