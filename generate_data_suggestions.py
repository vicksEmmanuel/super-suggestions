# %%


import csv
import pandas as pd
import shutil
from git import Repo
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import Language
from langchain.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import numpy as np
import sys
import uuid


repo_path = "repositories"
codebase_path = "codebases"


# %%
filename = "repo.csv"
repositories = []

with open(filename, mode="r") as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for row in csv_reader:
        repositories.append(row[0])


# %%
for repo in repositories:
    modify_https = repo.replace(":","_",1)
    try:
        Repo.clone_from(repo, to_path=f"{repo_path}/{modify_https}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Restructure python projects
# %%
jsSuffixes = [".js", ".ts"]
pySuffixes = [".py"]

textSuffixes = [
    # ".txt",
    ".md",
    ".rst",
    ".html",
    ".jsx",
    ".tsx",
]

otherLanguageSuffixes = [
    ".cpp",
    ".cs",
    ".java",
    ".php",
    ".go",
    ".swift",
    ".kt",
    ".dart",
    ".sh",
]

# %%
suffixes = pySuffixes + jsSuffixes + textSuffixes + otherLanguageSuffixes


# %%

for foldername, _, filenames in os.walk(repo_path):
    for filename in filenames:
        _, ext = os.path.splitext(filename)

        if ext in suffixes:
            source_file_path = os.path.join(foldername, filename)
            _, ext = os.path.splitext(filename)
            filename = f"{uuid.uuid4()}{ext}"

            target_file_path = os.path.join(codebase_path, filename)
            shutil.move(source_file_path, target_file_path)


# %%

jsLoaders = GenericLoader.from_filesystem(
    path=codebase_path,
    glob="*",
    suffixes=jsSuffixes,
    show_progress=True,
    parser=LanguageParser(language=Language.JS, parser_threshold=500),
)


pyLoaders = GenericLoader.from_filesystem(
    path=codebase_path,
    glob="**/*",
    suffixes=pySuffixes,
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)

textLoaders = GenericLoader.from_filesystem(
    path=codebase_path,
    glob="*",
    suffixes=textSuffixes,
    show_progress=True,
    parser=LanguageParser(parser_threshold=500),
)

allLoaders = {
    "jsLoaders": jsLoaders,
    "textLoaders": textLoaders,
    "pyLoaders": pyLoaders,
}

# Load documents using each loader
allLoaded = {
    "jsLoaders": allLoaders["jsLoaders"].load(),
    "textLoaders": allLoaders["textLoaders"].load(),
    "pyLoaders": allLoaders["pyLoaders"].load(),
}


def split_by_spaces(s, n=2):
    """
    Splits the input string s by the first n spaces.
    Returns a tuple of two strings:
    - The substring before the first n spaces.
    - The full input string.
    """
    parts = s.split(" ", n)
    if len(parts) > n:
        # If there are more than n parts, join the last parts.
        return (" ".join(parts[:n]), s)
    else:
        # If there are n or fewer parts, return the input string as both elements.
        return (s, s)


splitters = [50, 60, 80, 100, 150, 250, 370, 300, 350, 400, 450, 500, 550, 1000, 2000]


def createDataset(splitNumber, spaceNumber, indexValue):
    # Create splitters for each language
    allSplitters = {
        "jsLoaders": RecursiveCharacterTextSplitter.from_language(
            language=Language.JS, chunk_size=splitNumber, chunk_overlap=splitNumber
        ),
        "textLoaders": RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML, chunk_size=splitNumber, chunk_overlap=splitNumber
        ),
    }

    allTexts = {
        jsLoaders: allSplitters["jsLoaders"].split_documents(allLoaded["jsLoaders"]),
        textLoaders: allSplitters["textLoaders"].split_documents(
            allLoaded["textLoaders"]
        ),
    }

    inputText = []

    # Iterating over each key-value pair in the allTexts dictionary
    for loader, split_texts in allTexts.items():
        for index, text in enumerate(split_texts):
            text_content = (
                text.page_content
            )  # Assuming page_content is the actual text you want to process
            if (
                len(text_content.split()) > spaceNumber
            ):  # Only process texts with more than 2 words
                input_text, output_text = split_by_spaces(text_content, n=spaceNumber)
                inputText.append({"inputText": input_text, "outputText": output_text})

    data_path = f"llm_data/output_file_{indexValue}.csv"

    with open(data_path, "w", newline="", encoding="utf-8") as csvfile:
        print(data_path)
        csvwriter = csv.DictWriter(csvfile, fieldnames=["INPUT", "OUTPUT"])
        csvwriter.writeheader()

        for row in inputText:
            csvwriter.writerow({"INPUT": row["inputText"], "OUTPUT": row["outputText"]})

    print(f"Data has been written to {data_path}")


# %%

for index, splitter in enumerate(splitters):
    createDataset(splitNumber=splitter, spaceNumber=(index + 1) * 2, indexValue=index)


# %%

# DatasetDict({
#     train: Dataset({
#         features: ['conversation', 'summary', 'text'],
#         num_rows: 879
#     })
#     validation: Dataset({
#         features: ['conversation', 'summary', 'text'],
#         num_rows: 110
#     })
#     test: Dataset({
#         features: ['conversation', 'summary', 'text'],
#         num_rows: 110
#     })
# })
