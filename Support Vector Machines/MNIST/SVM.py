import pickle
import gzip
import time
import numpy as np
from sklearn import svm 

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)

def svm_baseline():
    print (time.strftime('%Y-%m-%d %H:%M:%S') )
    training_data, validation_data, test_data = load_data()
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03)
    clf.fit(training_data[0], training_data[1])
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print ("%s of %s test values correct." % (num_correct, len(test_data[1])))
    print (time.strftime('%Y-%m-%d %H:%M:%S'))
 
if __name__ == "__main__":
    svm_baseline()