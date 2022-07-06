import matplotlib.pyplot as plt
import sys
import pickle


path=sys.argv[1]

f=open(path, "rb")
losses=pickle.load(f)

train_loss=[]
for item in losses["train"]:
    train_loss.append( sum(item)/len(item) )

test_loss=[]
for item in losses["test"]:
    test_loss.append( sum(item)/len(item) )

plt.plot(train_loss, 'r')
plt.plot(test_loss, 'g')
plt.show()