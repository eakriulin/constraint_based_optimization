# Constraint Based Optimization

Training three neural networks considering constraints they impose on each other.

1. height -> age.
2. age -> weight.
3. height -> weight.

## Setup

Download and enter the project:

```zsh
git clone https://github.com/eakriulin/constraint_based_optimization.git
cd constraint_based_optimization
```

Create and activate the virtual environment

```zsh
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```zsh
pip3 install -r requirements.txt
```

## Run

```zsh
usage: main.py [-h] [--generate_dataset] [--height_age] [--age_weight] [--height_weight] [--constraint]

options:
  -h, --help          show this help message and exit
  --generate_dataset  If passed, the dataset will be generated
  --height_age        If passed, the height_age nn will be trained and evaluated
  --age_weight        If passed, the age_weight nn will be trained and evaluated
  --height_weight     If passed, the height_weight nn will be trained and evaluated
  --constraint        If passed, all three nns will be trained in constraint mode and evaluated
```

Example

```zsh
python3 main.py --constraint
```
