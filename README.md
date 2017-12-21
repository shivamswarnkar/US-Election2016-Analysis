# US Election 2016 Analysis
##

Please open this [project report](https://github.com/shivamswarnkar/US-Election2016-Analysis/blob/master/Project%20Report.pdf) for more detailed information. Following content only explains part of the project

### Abstract

This project analyzed US 2016 election county results with respect to county’s demographic features like population, population change, Black population, Hispanic population, age, education, income, poverty and density. 

Logistic regression and subset regression were used to rank all the combinations of features in order of prediction accuracy. The best accuracy from all tested logistic regression models was 91.43%, while population was the most successful single feature with 86.97% accuracy result. After analyzing the logistic regression, same data samples were used to train a Neural Network with 1000 hidden layers which gave the prediction accuracy of 93.05% in 200 epochs. 

In addition to training with all features, NN was separately also trained with best and worst feature combinations ranked from subset regression which gave 93.57% and 82.5% accuracy respectively. From result analysis, it was concluded that Democrats had higher chances of winning where Hispanic population, Black population and Education rates were higher, while Republicans had higher chances of winning with high poverty, high density and low education features.

### How to Use

Before running any files from the project, please make sure you have installed and downloaded all the required following dependencies and data files. 
* Numpy
* Pandas
* Sklearn
* Matplotlib
* Tensorflow 
* Keras 


Data Files
* votes-train.csv  [used to train the models]
* votes-test.csv      [used to test the models]

To reproduce the results, you should first run the [Logistic Regression.ipynb](https://github.com/shivamswarnkar/US-Election2016-Analysis/blob/master/Logistic%20Regression/Logistic%20Regression.ipynb) file, which will give you step by step results in form of printed strings and graphs. This file is well commented and should be easy to use. However, if you want, you can also you [Logistic Regression.py](https://github.com/shivamswarnkar/US-Election2016-Analysis/blob/master/Logistic%20Regression/LogisticRegression.py)  file to reproduce results as well. In the last sections of Logistic Regression.ipnb file, you can find several graphs which can help in visual analysis of features. 

To reproduce the results from Neural Network, you can run [NeuralNetTensor-flow.ipynb](https://github.com/shivamswarnkar/US-Election2016-Analysis/blob/master/Neural%20Network/NeuralNetTensor-flow.ipynb) file. Remember that this file hardwires the results from Logistic regression for best and worst features, therefore any changes in main data files will not update worst and best features in NN. Use of ipynb file is recommended for learning/understanding the project. 

The file [BackpropogationEntrop.m](https://github.com/shivamswarnkar/US-Election2016-Analysis/blob/master/Neural%20Network/Backpropogation%20Algorithm/BackpropogationEntrop.m) (matlab file) is an attempt at making N0-N1-N2 type of Neural Network from scratch. This file is independent from project; therefore, it can be used to create and train any N0-N1-N2 type of Neural Network. This implementation uses cross-entropy as cost function, sigmoid as transfer function and quadratic cost function to calculate error, which is used as convergence condition.

You can download [ProjectReport.pdf](https://github.com/shivamswarnkar/US-Election2016-Analysis/blob/master/Project%20Report.pdf) for more information. 

### Note: 

##### This project was built as a final project for the class Intro to Machine Learning taught by Professor Sundeep Rangan, at NYU Tandon School of Engineering.

All project artifacts (documents and code) were produced by Shivam Swarnkar. No previous project artifacts such as source codes were used in this project.



### Thank you. (I'll update README.me soon)


