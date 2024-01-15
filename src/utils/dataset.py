import random
import os
import csv
import numpy as np

# note: age in years, height range in cm, weight range in kg
age_groups_data = [
    (1, (72, 81), (9, 10.9)),
    (2, (83.8, 91.4), (10.9, 13.6)),
    (3, (92.7, 101.6), (13.2, 15.4)),
    (4, (100.3, 109.2), (15, 18.1)),
    (5, (106.7, 115.6), (16.8, 20.9)),
    (6, (113, 120.7), (19.1, 23.1)),
    (7, (119.4, 129.5), (20.9, 25.9)),
    (8, (125.7, 135.9), (22.7, 28.6)),
    (9, (132, 143.5), (24.9, 31.8)),
    (10, (137.2, 149.9), (27.2, 34.9)),
    (11, (143.5, 153.7), (29.5, 38.1)),
    (12, (147.3, 158.8), (31.8, 41.3)),
    (13, (151.1, 163.8), (34, 45.4)),
    (14, (154.9, 170.2), (36.3, 50.8)),
    (15, (158.8, 175.3), (38.6, 55.8)),
    (16, (162.6, 177.8), (40.8, 61.2)),
    (17, (165.1, 180.3), (43.1, 65.8)),
    (18, (166.4, 182.9), (45.4, 70.3)),
    (19, (167.6, 184.2), (47.6, 72.6)),
    (20, (167.6, 185.4), (49.9, 74.8))
]

def generate_age_data(age, height_range, weight_range, number_of_examples, data):
    for _ in range(number_of_examples):
        height = round(random.uniform(*height_range), 1)
        weight = round(random.uniform(*weight_range), 1)
        data.append((age, height, weight))

def write_data_to_file(filepath, data):
    with open(filepath, 'w') as file:
        writer = csv.writer(file)
        field = ["age", "height", "weight"]

        writer.writerow(field)
        for row in data:
            writer.writerow(row)

def generate_dataset():
    train_data = []
    test_data = []
    for age, height_range, weight_range in age_groups_data:
        generate_age_data(age, height_range, weight_range, number_of_examples=25, data=train_data)
        generate_age_data(age, height_range, weight_range, number_of_examples=5, data=test_data)

    write_data_to_file(os.path.join('dataset', 'train', 'age_height_weight_dataset.csv'), train_data)
    write_data_to_file(os.path.join('dataset', 'test', 'age_height_weight_dataset.csv'), test_data)
