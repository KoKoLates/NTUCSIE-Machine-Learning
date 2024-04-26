# Hw1: Perceptron Learning Algorithm
`perceptron` 、 `bad data` 、 `hoeffding's inequality`
## Assignment
* [homework 01](./hw1.pdf)
* [homework report](./hw1_report.pdf)

## Usage
running with help flag `-h` for execuation instruction.
```
usage: hw1.py [-h] -f F -q Q

options:
    -h, --help  show this help message and exit
    -f F        the file path of data
    -q Q        the question number from 16 to 20
```
both arguments (`f` and `q`) are required, the former is to indicate the file path of training and testing data. And the latter one is for the question number in programming part.


Running problem 18 for example:
```
python hw1.py -f data/train.dat -q 18
```
