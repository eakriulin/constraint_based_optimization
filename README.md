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

## Dataset

Dataset is generated randomly based on the predefined value ranges for each age group (age, height range in cm, weight range in kg). Value ranges are selected in such a way that it's possible to separate one age group from another.

The dataset is split into two parts: 80% for training and 20% for testing.

## Data Normalization

To achieve numerical stability, input values are normalized before going into a neural network.

$\text{Normalized value} = \frac{x - \mu}{\sigma}$

- $x$ is the original value of an input.
- $\mu$ is the mean of the input values.
- $\sigma$ is the standard deviation of the input values.

## Loss Functions

### Individual learning

$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$

- $n$ — number of values
- $Y_i$ — predicted value
- $\hat{Y}_i$ — target value

### Constraint learning

$UnifiedLoss = \alpha_1 \cdot MSE_{NN1} + \alpha_2 \cdot MSE_{NN2} + \alpha_3 \cdot MSE_{NN3} + \alpha_c \cdot LossConstraint$

- $NN_1$ — neural network 1
- $NN_2$ — neural network 2
- $NN_3$ — neural network 3
- $\alpha_1, \alpha_2, \alpha_3, \alpha_c$ — arbitrary coefficients

$LossConstraint = \frac{1}{n} \sum_{i=1}^{n} (NN_3(A_i) - NN_2(B_i))^2$

- $n$ — number of values
- $A_i$ — input value for neural network 3
- $B_i$ — input value for neural network 2

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
4. In the experiments, it was found that lowering $\alpha_2$ improved performance. This is likely because it slows down the training of $NN2$, allowing it to "wait" until $NN1$ starts producing accurate predictions.
