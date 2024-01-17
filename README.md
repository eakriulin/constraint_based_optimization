# Constraint Based Optimization

Training three neural networks considering constraints they impose on each other.

1. height -> age
2. age -> weight
3. height -> weight

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
usage: main.py [-h] [--generate_dataset] [--height_age] [--age_weight] [--height_weight] [--independent] [--constraint]

options:
  -h, --help          show this help message and exit
  --generate_dataset  If passed, the dataset will be generated
  --height_age        If passed, the height_age nn will be trained and evaluated
  --age_weight        If passed, the age_weight nn will be trained and evaluated
  --height_weight     If passed, the height_weight nn will be trained and evaluated
  --independent       If passed, all three nns will be trained independently and evaluated
  --constraint        If passed, all three nns will be trained in constraint mode and evaluated
```

Example

```zsh
python3 main.py --constraint
```

## Loss Functions

### Individual learning

$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$

* $n$ — number of values
* $Y_i$ — predicted value
* $\hat{Y}_i$ — target value

### Constraint learning

$UnifiedLoss = \alpha \cdot MSE_{NN1} + \beta \cdot MSE_{NN2} + \gamma \cdot MSE_{NN3} + \lambda \cdot LossConstraint$

* $NN_1$ — neural network 1
* $NN_2$ — neural network 2
* $NN_3$ — neural network 3
* $\alpha, \beta, \gamma, \lambda$ — arbitrary coefficients

$LossConstraint = \frac{1}{n} \sum_{i=1}^{n} (NN_3(A_i) - NN_2(B_i))^2$

* $n$ — number of values
* $A_i$ — input value for neural network 3
* $B_i$ — input value for neural network 2

## Evaluation Results

### Individual learning

1. height -> age: MSE = 0.646
2. age -> weight: MSE = 1.096
3. height -> weight: MSE = 2.266

### Independent learning

1. height -> weight: MSE = 2.330
2. height -> age -> weight: MSE = 74.914

### Constraint-based learning

1. height -> weight: MSE = 2.351
2. height -> age -> weight: MSE = 2.274

## Comments

1. Individual learning demonstrates the best results assuming the neural networks will always operate separately and independently.
2. Independent learning shows poor performance in the interconnected height-age-weight prediction, indicating that independent training cannot effectively capture the dependencies between the models.
3. Constraint-based learning significantly improves the interconnected height-age-weight prediction. This suggests that imposing constraints that acknowledge the relationships between height, age, and weight leads to more accurate results.
