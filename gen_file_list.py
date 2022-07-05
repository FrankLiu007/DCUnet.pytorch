import os
import glob
import random
import pickle
import sys

def get_file_list(path):
    rfs=[]
    z_cmps=[]
    result=[]
    for rf in  glob.glob( os.path.join(path, "china/2.5/*sac*") ) :
        #z_cmp=rf[:-11]+"BHZ.0.sac"
        z_cmp=os.path.basename(rf)[:-11]+"BHZ.0.sac"
        z_cmp=os.path.join(path, "china/resample_cut", z_cmp)
        if not os.path.exists( z_cmp ):
            print("file ", z_cmp, "not found")
            continue
        result.append((z_cmp, rf))



    for rf in  glob.glob( os.path.join(path, "us/2.5/*sac*") ) :
        z_cmp=os.path.basename(rf)[:-11]+"BHZ.0.sac"
        z_cmp=os.path.join(path, "us/resample_cut", z_cmp)
        if not os.path.exists( z_cmp ):
            print("file ", z_cmp, "not found")
            continue

        result.append((z_cmp, rf))

    return result

def split_dataset(file_list, percent):
    data={}
    train_set={}
    test_set={}
    for z_cmp, rf in file_list:
        kk=os.path.basename(z_cmp).split(".")
        stnm=kk[0]+"."+kk[1]
        if stnm in data:
            data[stnm].append((z_cmp, rf))
        else:
            data[stnm]=[(z_cmp, rf)]


    for key, value in  data.items():
        n=round(len(value)*percent)

        test_set[key]= random.sample(value, n)
        for  item in  test_set[key]:
            value.remove(item)
    train_set=data
    return train_set, test_set
def write_data(path, obj):
    f=open(path, "wb")
    pickle.dump(obj, f)
    f.close()

def main(path):
    file_list=get_file_list(path)
    train, test=split_dataset(file_list, 0.1)
    write_data(os.path.join(path, "dataset.lst"), {"train":train, "test":test})
if __name__ == '__main__':
    main(sys.argv[1])