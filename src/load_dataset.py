from os import path

from datasets import load_dataset

DATASET_PATH = "medical_questions_pairs"
PATH_TO_DATASET = path.join("data", "medical_questions_pairs.csv")


def main():
    dataset = load_dataset(path=DATASET_PATH, split="train")
    dataset.to_pandas().to_csv(path_or_buf=PATH_TO_DATASET, index=False)


if __name__ == "__main__":
    main()
