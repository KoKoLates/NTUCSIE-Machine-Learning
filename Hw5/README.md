# Soft-Margin Support Vector Machine
`support vector machine` 、 `kernel models`

## Assignments
* [homework 5](./hw5.pdf)
* [homework report](./hw5_report.pdf)

## Usage

This part require [`libsvm`](https://github.com/cjlin1/libsvm/tree/master) packages. User could clone the packages from the github directly with the git command as below:

```bash
git clone https://github.com/cjlin1/libsvm.git
```

After the `libsvm` and dataset are both setup, the folder structure will be like the figure shown below. Be careful that the path and name of files are proper and correct, otherwise, the program will raise some error due to the relative path or libraries missing issue.

```
Hw5
├── data
│   ├── train.dat
│   └── test.dat
├── libsvm
|   ├── ...
|   ├── python/libsvm
|   |   ├── __init__.py   
│   |   ├── svm.py
|   |   └── svmutil.py
│   ├── svm.cpp
|   └── Makefile
├── hw5.py 
└── README.md
```

User could read the execution instruction by running with help flag `-h` of the program.

```
usage: hw5.py [-h] [-q Q]

options:
    -h, --help  show this help message and exit
    -q Q        select specific quesion for execution.
```

You can use the `-q` flag to execute indicate question in problem list that contain questions from 15 to 20 of programming part of assignment for testing or aquire the correspoding results.

```bash
python hw5.py -q 18 # execute only question 18
```

Or just running `hw5.py` in defualt for executing all questions sequencely in the problem list.

```bash
python hw5.py # execute all questions
```
