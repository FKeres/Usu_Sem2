import numpy as np
from sklearn.tree import DecisionTreeRegressor

class Bag:
    def __init__(self, num_of_bags, length_of_subtrain):
        self.num_of_bags = num_of_bags
        self.length_of_subtrain = length_of_subtrain
        self.models = []
    
    def fit(self, input, output, max_depth=None, min_samples_split=2):
        for i in range(self.num_of_bags):
            index = np.random.choice(len(input), size=int(self.length_of_subtrain/100 * len(input)), replace=True)
            input_sample = input.iloc[index]
            output_sample = output.iloc[index]
            
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
            model.fit(input_sample, output_sample)
            self.models.append(model)
    
    def predict(self, input):
        predictions = np.array([model.predict(input) for model in self.models])
        return np.mean(predictions, axis=0)
