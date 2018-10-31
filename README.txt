NeuralNet
This file contains :
1.NeuralNet code
2.report of our results 
3.theoretical numerical questions.

Requirements
"NeuralNet.py" require the following to run:
*python 3.5+
*Pycharm with these Project Interpreters: numpy, pandas.


Usage
*The training dataset is "iristraining.csv".
*The testing dataset is "iristest.csv".
*Put the "iristraining.csv" and "iristest.csv" under the same folder with "NeuralNet.py" 
*If you want to test different activation functions, please change "Activation_Function_Name"
####
if __name__ == "__main__":
    
neural_network = NeuralNet("iristraining.csv","Activation_Function_Name")
    
neural_network.train("Activation_Function_Name")
    
testError = neural_network.predict("iristest.csv","Activation_Function_Name")
####