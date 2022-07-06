import matplotlib.pyplot as plt
import sys
import pickle


path=sys.argv[1]

f=open(path, "rb")
losses=pickle.load(f)
f.close()

train_loss=[]
for item in losses["train"]:
    train_loss.append( sum(item)/len(item) )

test_loss=[]
relatives=[]
for item,relative in losses["test"]:
    test_loss.append( sum(item)/len(item) )
    relatives.append(sum(relative)/len(relative))

plt.plot(train_loss, 'r')
plt.plot(test_loss, 'g')
plt.plot(relatives)
plt.show()


