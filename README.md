# Thesis Similarity Checker SVM V2

This repository contains a project for checking the similarity of thesis documents using Support Vector Machine (SVM) models. The project is designed to assist in identifying and analyzing similarities between academic theses.

## Features

- **SVM-based Similarity Checking**: Utilizes Support Vector Machine models to perform document similarity checks.
- **Document Preprocessing**: Includes functions for text preprocessing such as tokenization, stemming, and stop-word removal.
- **Similarity Metrics**: Implements various similarity metrics to evaluate the similarity between documents.
- **Extensive Testing**: Contains unit tests to ensure the accuracy and reliability of the similarity checking process.

## Installation

To install and set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/delosreyesjohnpaul/Thesis-Similarity-Checker-SVM-V2.git
    cd Thesis-Similarity-Checker-SVM-V2
    ```

2. Set up a virtual environment and install dependencies:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

To use the similarity checker, you can run the main script with your documents as input. Example usage:

```sh
python main.py --input1 document1.txt --input2 document2.txt
