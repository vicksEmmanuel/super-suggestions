# %%

import csv
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

filename = "llm_data/output_file"
number_of_files = 8

repositories = []

# %%

for i in range(0, number_of_files):
    with open(f"{filename}_{i}.csv", mode="r") as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            repositories.append({"input": row[0], "output": row[1]})


df = pd.DataFrame(repositories)

# %%
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# # Create Dataset objects for each subset
# train_dataset = Dataset.from_pandas(train_df)
# valid_dataset = Dataset.from_pandas(valid_df)
# test_dataset = Dataset.from_pandas(test_df)

# # Create a DatasetDict to hold the Dataset objects
# dataset_dict = DatasetDict(
#     {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
# )

# print(dataset_dict)


# # %%
# dataset_dict.save_to_disk("llm_data/dataset")

# %%

print("Saving to csv")

train_df.to_csv("llm_data/train.csv", index=False)
valid_df.to_csv("llm_data/validation.csv", index=False)
test_df.to_csv("llm_data/test.csv", index=False)

print("Done")

# %%
