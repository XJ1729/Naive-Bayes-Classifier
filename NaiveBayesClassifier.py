# Naive Bayes to predict movie ratings on a scale of 1-5
import numpy as np
import json
from collections import Counter

# +
# Data Loading

training_path = '/autograder/grade/private/t-r-a-i-n.json'

try:
    # Open the JSON file in read mode ('r')
    with open(training_path, 'r') as f:
        # Load the JSON data from the file
        t_data = json.load(f)
#     print("JSON data loaded successfully:")
except FileNotFoundError:
    print(f"Error: The file '{training_path}' was not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{training_path}'. Check file format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    


# +
validation_path = '/autograder/grade/private/t-e-s-t.json'

try:
    with open(validation_path, 'r') as f:
        v_data = json.load(f)
#     print("JSON data loaded successfully:")
except FileNotFoundError:
    print(f"Error: The file '{validation_path}' was not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{validation_path}'. Check file format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
# -

# Check the data types
# print(type(v_data))
# print(type(v_data[0]))


# P(target | x1, ... xn) = P(target) * P(x1 | target) * ... * P(xn | target) / P(X)
class NaiveBayesClassifier:
    
    # what we are predicting
    target = ["rating"]
    
    # separates into two lists: a list of variables and list of targets
    def separate_variables_from_target(self, dataset):
        X = []
        # our goal values (the ratings)
        Y = []
        # for each dictionary in dataset
        for dic in dataset:
            target_list = [dic[k] for k in self.target if k in dic]
            Y.append(target_list[0])
            
            variables = {k: v for k, v in dic.items() if k not in self.target}
            X.append(variables)
    
        return X, Y
    
    # this function 
    def separate_by_class(self, X, Y):
        t_class = {}
        for variables, target in zip(X, Y):
            if target not in t_class:
                t_class[target] = []
            t_class[target].append(variables)
        return t_class
    
    
    # training
    
    def calculuate_variable_probabilities(self, data):
        variable_probabilities = {}
        for clas, rows in data.items():
            variable_probabilities[clas] = {}
            variable_keys = rows[0].keys()  
            for key in variable_keys:
                variable_values = [row[key] for row in rows]
                value_counts = Counter(variable_values)
                total_count = len(rows)
                probabilities = {value: count / total_count for value, count in value_counts.items()}
                variable_probabilities[clas][key] = probabilities
        return variable_probabilities
    
    # Calculating P(target)/class prior probability
    
    def calculate_p_targets(self, p):
        counts = Counter(p)
        total = len(p)
        p_targets = {c: counts[c] / total for c in counts}
        return p_targets
    
    # prediction
    
    def predict(self, tester, p_targets, variable_probabilities):
        probabilities = {}
        for clas in p_targets:
            # P(target)
            probabilities[clas] = p_targets[clas]
            # Multiply by the probability of each feature value given the class
            for variable, value in tester.items():
                if variable in variable_probabilities[clas]:
                    if value in variable_probabilities[clas][variable]:
                        probabilities[clas] *= variable_probabilities[clas][variable][value]
                    else:
                        # smoothing for unseen feature values
                        probabilities[clas] *= 1e-6
                else:
                    probabilities[clas] *= 1e-6
        # Return the class with the highest probability (classification)
        return max(probabilities, key=probabilities.get)
    

# +
# more preprocessing: removal of unecessary features such as user_id, item_id, user_occupation, user_zip_code, movie_title, movie_genres, movie_description, user_review

features_to_remove = ["user_id", "item_id", "movie_genres", "movie_description", "user_review"]

clean_t = [{k: v for k, v in d.items() if k not in features_to_remove} for d in t_data]

clean_v = [{k: v for k, v in d.items() if k not in features_to_remove} for d in v_data]


n = NaiveBayesClassifier()

X,Y = n.separate_variables_from_target(clean_t)
v_x, v_y = n.separate_variables_from_target(clean_v)


# +
t_by_class = n.separate_by_class(X,Y)
# print(t_by_class[1])

p_targets = n.calculate_p_targets(Y)
# print(priors)

variable_probs = n.calculuate_variable_probabilities(t_by_class)
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


