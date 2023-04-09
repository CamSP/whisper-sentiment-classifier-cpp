import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

datos_train=np.loadtxt("../results/train_results.csv", delimiter=",", dtype=float)
Loss_train = datos_train[:,0]
Acc_train = datos_train[:,1]


datos_test=np.loadtxt("../results/test_results.csv", delimiter=",", dtype=float)
Loss_test = datos_test[:,0]
Acc_test = datos_test[:,1]

plt.plot(Loss_train, label="Train")
plt.plot(Loss_test, label="Test")
plt.xlabel("Epoch")
plt.legend()
plt.title("Loss")
plt.grid()
plt.savefig("../graphs/Loss.pdf")

plt.figure()
plt.plot(Acc_train, label="Train")
plt.plot(Acc_test, label="Test")
plt.ylim([0, 1])
plt.xlabel("Epoch")
plt.legend()
plt.grid()
plt.title("Acc")
plt.savefig("../graphs/Acc.pdf")
