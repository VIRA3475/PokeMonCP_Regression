import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib


df = pd.read_csv("data/pokemon.csv")
# print(df)
# df.plot.scatter('cp','newcp')
# plt.show()
random_df=df.sample(frac=1)
train_data=random_df[20:31]
train_data.index = range(len(train_data))
test_data=random_df[:10]
test_data.index = range(len(test_data))



# print(train_data,"\n\n",test_data)
X_train=train_data.cp 
Y_train=train_data.newcp
# print(X_train)
# print(len(X_train))
X_test=test_data.cp 
Y_test=test_data.newcp
# print(X_train[1])




##y=b+w*x
b = -120 # initial b
w = -4 # initial w
lr = 0.0000001 # learning rate
lda=100 # lambda
iteration = 10000 # 迭代次數



# Iterations
for i in range(iteration):
    
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(X_train)):   
        b_grad =b_grad-2.0*(Y_train[n] - b - w*X_train[n])*1.0
        #b_grad =b_grad-2.0*(Y_train[n] - b - w*X_train[n])*1.0+1  # Regularization
        print("b_grad: ",b_grad)
        w_grad =w_grad-2.0*(Y_train[n] - b - w*X_train[n])*X_train[n]
        #w_grad =w_grad-2.0*(Y_train[n] - b - w*X_train[n])*X_train[n]+2*lda*w  # Regularization
        print("w_grad: ",w_grad)
    
    # b_lr = b_lr + b_grad**2
    # w_lr = w_lr + w_grad**2
    
    # Update parameters.
    b = b - lr * b_grad
    print("b: ",b)
    w = w - lr * w_grad
    print("w: ",w)
    # b = b - lr/np.sqrt(b_lr) * b_grad 
    # w = w - lr/np.sqrt(w_lr) * w_grad


    



aaa=plt.scatter(X_train,Y_train)
aaa=plt.plot(X_train,b+w*X_train)

bbb=plt.scatter(X_test,Y_test)
bbb=plt.plot(X_test,b+w*X_test)
plt.show()

