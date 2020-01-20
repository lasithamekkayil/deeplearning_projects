import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

input_file = pd.read_csv('student_data.csv')
#input_file[:10]

#perform one hot encoding for rank
one_hot_data = []
one_hot_data = pd.concat([input_file, pd.get_dummies(input_file['rank'], prefix = 'rank')], axis =1)
one_hot_data = one_hot_data.drop('rank', axis=1)
#one_hot_data[:10]

#Normalise the gre and gpa
normalised = one_hot_data[:]
normalised['gre'] = normalised['gre']/input_file['gre'].max()
normalised['gpa'] = normalised['gpa']/input_file['gpa'].max()
#normalised[:10]

#Splitting the data into training and testing
sample = np.random.choice(normalised.index, size = int(len(normalised)*0.9), replace = False)                   
train_data, test_data = normalised.iloc[sample], normalised.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
#print(train_data[:10])
#print(test_data[:10])

#Split the data into features and targets
features = train_data.drop('admit', axis =1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis =1)
targets_test = test_data['admit']

#Training the 2 layer network
def sigmoid(x):
    return (1/(1+np.exp(-x)))
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def error_term_formula(x, y, output):
    return (y-output)*sigmoid_prime(x)

#Neural network hyperparameters
epochs = 1000
learning_rate = 0.5

#Training
def train_nn(features, targets, epochs, learning_rate):
    np.random.seed(42)
    n_records, n_features = features.shape
    last_loss = None
    
    weights = np.random.normal(scale = 1/n_features**0.5 , size = n_features)
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x,y in zip(features.values, targets):
            output = sigmoid(np.dot(x,weights))
            error = error_formula(y,output)
            error_term = error_term_formula(x,y,output)
            del_w += error_term*x
        weights += learning_rate * del_w/n_records
        
        if e % (epochs/10) ==0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out-targets)**2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:      
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
weights = train_nn(features, targets, epochs, learning_rate)
        

# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
    