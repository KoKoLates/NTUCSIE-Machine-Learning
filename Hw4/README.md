# Regularized Logistic Regression
`regularization` 、 `validation` 、 `learning principles`

## Assignments
* [homework 4](./hw4.pdf)
* [homewrok report](./hw4_report.pdf)

## Usage

This part require [`liblinear`](https://github.com/cjlin1/liblinear) packages for model training and prediction. Ones could clone the packages from the github directly with the git command as below.
```bash
git clone https://github.com/cjlin1/liblinear.git
```
After the `liblinear` and dataset setup, the folder structure will be like the figure shown below. Be careful that the path and name of files are proper and correct, otherwise, the program will raise some error due to the relative path or libraries missing issue.
```
Hw4
├── data
│   ├── train.dat
│   └── test.dat
├── liblinear
|   ├── ...
|   ├── python/lib
|   |   ├── __init__.py   
│   |   ├── liblinear.py
|   |   └── liblinearutil.py
│   ├── train.c
|   └── Makefile
├── hw4.py 
└── README.md

```
By running with help flag `-h` of the program for the execution instruction.
```
usage: hw4.py [-h] [-q Q]

options:
    -h, --help  show this help message and exit
    -q Q        select specific quesion for execution.
```
You can use the `-q` flag to execute indicate question in problem list that contain questions from `16` to `20` of programming part of assignment for testing or aquire the correspoding results.

``` shell
python hw4.py -q 18 # execute only question 18
```

Or just running `hw4.py` in defualt for executing all questions in the problem list.

``` shell
python hw4.py # execute all questions
```
