import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([20,50,35,67,25,41,15,4,32,25,6,10]).reshape(-1,1)
scores = np.array([56,85,42,93,47,82,45,78,56,63,57,17]).reshape(-1,1)

model = LinearRegression()
model.fit(time_studied,scores)

print(model.predict(np.array([56]).reshape(-1,1)))

plt.scatter(time_studied,scores)
plt.plot(np.linspace(0,70,100).reshape(-1,1),model.predict(np.linspace(0,70,100).reshape(-1,1)),'r')
plt.ylim(0,100)
plt.xlabel("time_studied")
plt.ylabel("scores")
plt.savefig("LRplot.png")
plt.show()


#Testing
time_train, time_test, score_train, score_test = train_test_split(time_studied,scores,test_size=0.2)
print(model.score(time_test,score_test))