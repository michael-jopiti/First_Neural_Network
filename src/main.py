from model import Model 
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split 
import os

def process_df(df):
    df["variety"] = df["variety"].replace("Setosa", 0.0)
    df["variety"] = df["variety"].replace("Versicolor", 1.0)
    df["variety"] = df["variety"].replace("Virginica", 2.0)
    
    return df

def plot_losses(epochs, losses, name="src/plots/loss.png"):
    plt.plot(range(epochs), losses)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.savefig(name)
    plt.close()

def main():
    print("This is my very first Neural Network :D")

    torch.manual_seed(14)

    model = Model()
    #  We need to create a model
    # let model = Model()
    
    url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    iris = pd.read_csv(url)
    #Replace the species names with float values (0.0, 1.0, 2.0)
    data = process_df(iris)

    # Let's create a training set 
    # to do so, we need to drop variety (we only need ot keep the features, not the tags)
    X = data.drop("variety", axis=1)
    # The column dropped represents the correct prediction, therefore we need to store it in another, separate, dataframe
    y = data["variety"]
    
    # ultimetly, we need to transform the dataframe into numpy array
    X = X.values
    y = y.values

    #Let's subdivise our set(s) into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

    #But we need them into tensors, so let's transform them!
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test= torch.LongTensor(y_test)
     
    # a function to estimate the error is needed!
    # we're gonna use the Adam Optimizer with an arbitrary learning rate (the smaller, the slower the training phase)
    criterion = nn.CrossEntropyLoss()
    # In plain words, the optimizer goes through all model parameters (layers) once per epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Let's finally train the model!
    # Let's define how many epochs we want the model to go through and let's store the error for future plotting
    epochs = 100
    losses = []


    for i in range(epochs):
        # let's move from input through all the layers
        y_pred = model.forward(X_train)
        # Let's calculate the loss (should be exponentially decaying)
        loss = criterion(y_pred, y_train)
        losses.append(loss.detach().numpy())  
        # print the loss at each x epochs (I'll choose 10)
        if  i % 10 == 0:
            print(f'Epoch: {i}, loss: {loss}')
            
        # We now need backpropagation to "learn", which means readjusting the wights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # plot and store the image for checking the loss decrease through the different epochs
    plot_losses(epochs, losses)
    
    #Now that the model is trained, we turn off backpropagation (learning function), and we pass through the ANN to see actual predictions
    with torch.no_grad():  # Disable gradient calculation
    	correct = 0
    	y_eval = model(X_test)  # Forward pass for the entire test set
    	loss = criterion(y_eval, y_test)  # Compute loss for the batch
    
    # Compare predictions
    	_, predicted = torch.max(y_eval, 1)  # Get predicted class indices
    	correct = (predicted == y_test).sum().item()  # Count correct predictions
    
    # Print predictions and actual values
    	for i in range(len(X_test)):
        	print(f'Prediction: {predicted[i].item()}, Actual: {y_test[i].item()}')
    
    print(f'Correct predictions: {correct}/{len(X_test)}')
if __name__ == "__main__":
    main()
