import pandas as pd
import os
import random
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_split.py <metadata.csv> <output_folder>")
        sys.exit(1)

    metadata_path = sys.argv[1]
    out_folder = sys.argv[2]

    os.makedirs(out_folder, exist_ok=True)

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Keep only entries that have a text_code (just cleaner)
    df = df[df["text_code"].notnull()]

    # Extract unique subjects
    subjects = sorted(df["subject"].unique())

    # Shuffle subjects for random split
    random.seed(42)
    random.shuffle(subjects)

    # 80% train, 10% val, 10% test
    n = len(subjects)
    train_cut = int(0.8 * n)
    val_cut = int(0.9 * n)

    train_subjects = set(subjects[:train_cut])
    val_subjects = set(subjects[train_cut:val_cut])
    test_subjects = set(subjects[val_cut:])

    # Assign split to each row
    def assign_split(sub):
        if sub in train_subjects:
            return "train"
        elif sub in val_subjects:
            return "val"
        else:
            return "test"

    df["split"] = df["subject"].apply(assign_split)

    # Save output
    out_csv = os.path.join(out_folder, "metadata_with_split.csv")
    df.to_csv(out_csv, index=False)

    print("Split completed!")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects:   {len(val_subjects)}")
    print(f"Test subjects:  {len(test_subjects)}")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
