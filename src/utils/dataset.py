import random
import os
import csv

# note: age in years, height range in cm, weight range in kg
age_groups_data = [
    (1,  (72.0,  81.0),  (9.0, 10.9)),
    (2,  (83.8,  91.4),  (11.1, 13.6)),
    (3,  (92.7,  101.6), (13.8, 15.4)),
    (4,  (102.3, 109.2), (15.6, 18.1)),
    (5,  (109.7, 115.6), (18.3, 20.9)),
    (6,  (116.0, 120.7), (21.1, 23.1)),
    (7,  (121.4, 129.5), (23.3, 25.9)),
    (8,  (130.7, 135.9), (26.1, 28.6)),
    (9,  (136.0, 143.5), (28.8, 31.8)),
    (10, (144.2, 149.9), (32.0, 34.9)),
    (11, (150.5, 153.7), (35.1, 38.1)),
    (12, (154.3, 158.8), (38.3, 41.3)),
    (13, (159.1, 163.8), (41.5, 45.4)),
    (14, (164.9, 170.2), (45.6, 50.8)),
    (15, (171.8, 175.3), (60.0, 62.8)),
    (16, (175.6, 177.8), (63.0, 65.0)),
    (17, (178.1, 180.3), (67.0, 69.0)),
    (18, (181.4, 182.9), (71.0, 73.0)),
    (19, (183.6, 184.2), (75.0, 77.0)),
    (20, (185.6, 185.4), (79.0, 81.0))
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
