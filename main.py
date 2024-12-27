import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from Bag import Bag

def bagging_mean_squared_error(i_train, i_test, o_train, o_test, length_of_subtrain, num_of_bags, num_iterations, max_depth=None, min_samples_split=2):
    mse_list = []
    
    for i in range(num_iterations):
        mse_bagging = 0
        bagging_model = Bag(num_of_bags=num_of_bags, length_of_subtrain=length_of_subtrain)
        bagging_model.fit(i_train, o_train, max_depth=max_depth, min_samples_split=min_samples_split)

        predicted = bagging_model.predict(i_test)
        mse_bagging = mean_squared_error(o_test, predicted)
        
        
        mse_list.append(mse_bagging)
    
    return np.mean(mse_list)

data = pd.read_csv('bag_6.csv')

data = pd.get_dummies(data)

input = data.drop("price", axis=1)
output = data["price"]

i_train, i_test, o_train, o_test = train_test_split(input, output, test_size=0.2, random_state=1)

subtrain_lengths = [40, 50, 60, 70]
num_of_bags_list = [5, 10, 15, 50, 100]

for num_of_bags in num_of_bags_list:
    for subtrain_length in subtrain_lengths:
        mse = bagging_mean_squared_error(
            i_train, 
            i_test, 
            o_train, 
            o_test, 
            subtrain_length, 
            num_of_bags, 
            1000)
        print(f"Mean Squared Error length of subtrain = {subtrain_length} num of bags {num_of_bags}: {mse}")

max_depth_values = [3, 5, 10, None]
min_samples_split_values = [2, 5, 10, 20]

for max_depth in max_depth_values:
    for min_samples_split in min_samples_split_values:
        mse = bagging_mean_squared_error(
            i_train, i_test, o_train, o_test, 
            length_of_subtrain=60, 
            num_of_bags=50, 
            num_iterations=1000, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split
        )
        print(f"Mean Squared Error length of subtrain = 60, num of bags = 50, max_depth = {max_depth}, min_samples_split = {min_samples_split}: {mse}")