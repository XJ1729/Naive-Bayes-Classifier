# Naive Bayes to predict movie ratings on a scale of 1-5
import sys
import json
import numpy as np
from collections import Counter

# +
# Data Loading

if len(sys.argv) != 3:
    sys.exit(1)

# training_path = 'training.json'
training_path = sys.argv[1]

try:
    # Open the JSON file in read mode ('r')
    with open(training_path, 'r') as f:
        # Load the JSON data from the file
        t_data = json.load(f)
#     print("JSON data loaded successfully:")
except FileNotFoundError:
    print(f"Error: The file '{training_path}' was not found.")
    


# +
# validation_path = 'validation.json'

validation_path = sys.argv[2]

try:
    with open(validation_path, 'r') as f:
        v_data = json.load(f)
#     print("JSON data loaded successfully:")
except FileNotFoundError:
    print(f"Error: The file '{validation_path}' was not found.")
# -

# Check the data types
# print(type(v_data))
# print(type(v_data[0]))


# P(target | x1, ... xn) = P(target) * P(x1 | target) * ... * P(xn | target) / P(X)
class NaiveBayesClassifier:
    
    target = "rating" # what we are predicting
    
    def remove_unnecessary(self, rem_list, orig):
        return [{k: v for k, v in dic.items() if k not in rem_list} for dic in orig]
    
    # separates into two lists: a list of variables and list of targets
    def separate_variables_from_target(self, dataset):
        X = []
        Y = []
        
        for dic in dataset:
            Y.append(dic['rating'])
            variables = {k: v for k, v in dic.items() if k != self.target}
            X.append(variables)
    
        return X, Y
    
    # separates into target classes for P(xi | target) calculation
    def separate_by_class(self, X, Y):
        t_class = {}
        for variables, target in zip(X, Y):
            if target not in t_class:
                t_class[target] = []
            t_class[target].append(variables)
        return t_class
    
    
    # training
    
    def calculate_variable_probabilities(self, t_class):
        var_probs = {}
        for clas, varz in t_class.items():
            var_probs[clas] = {}
            vkeys = varz[0].keys() 
            for key in vkeys:
                vvals = [var[key] for var in varz]
                vcount = Counter(vvals)
                tct = len(varz)
                probs = {value: count / tct for value, count in vcount.items()}
                var_probs[clas][key] = probs
        return var_probs
    
    # for calculating P(target)/class prior probability
    
    def calculate_p_targets(self, p):
        cts = Counter(p)
        total = len(p)
        p_targets = {c: cts[c] / total for c in cts}
        return p_targets
    
    # prediction
    
    def predict(self, tester, p_targets, var_probs):
        probs = {}
        for clas in p_targets:
            # P(target)
            probs[clas] = p_targets[clas]
            # Multiply by P(xi | target) 
            for var, value in tester.items():
                if var in var_probs[clas] and value in var_probs[clas][var]:
                    probs[clas] *= var_probs[clas][var][value]
                else:
                    probs[clas] *= 1e-12 # probability not equal to 0
        # Return the class with the highest probability (classification)
        return max(probs, key=probs.get)
    

# +
n = NaiveBayesClassifier()


# more preprocessing: removal of unnecessary features such as user_id, item_id, user_occupation, user_zip_code, movie_title, movie_genres, movie_description, user_review

remove = ["user_id", "item_id", "user_age", "movie_genres", "movie_description", "user_review"]

clean_t = n.remove_unnecessary(remove, t_data)

clean_v = n.remove_unnecessary(remove, v_data)



X,Y = n.separate_variables_from_target(clean_t)
v_x, v_y = n.separate_variables_from_target(clean_v)


# +
t_by_class = n.separate_by_class(X,Y)
# print(t_by_class[1])

p_targets = n.calculate_p_targets(Y)
# print(priors)

variable_probs = n.calculate_variable_probabilities(t_by_class)
# print(feature_probs[5])

# test = v_x[0]
# print(model.predict(test, priors, feature_probs))
# print(v_y[0])

# count = 0
# for row in range(len(v_x)):
#     if model.predict(v_x[row], priors, variable_probs) == v_y[row]:
#         count += 1
# print(count/len(v_x))

for row in range(len(v_x)):
    print(n.predict(v_x[row], p_targets, variable_probs))
# -


