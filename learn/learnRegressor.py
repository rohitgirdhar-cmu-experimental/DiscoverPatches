from sklearn import preprocessing, svm
import numpy as np
import os, pickle
import pdb

featdir = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/features/CNN_fc7_text'
imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/ImgsList.txt'
trainlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/split/TestList.txt'
scoresdir = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/matches_scores'
cachedir = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/learnGoodPatches/scratch'
RAND_SEL = 800 # randomly select these many from each image

def main():
  ## read the data
  data, labels = readData()
#  data = data[np.arange(2000)]
#  labels = labels[np.arange(2000)]
  data = preprocessing.scale(data, axis=0) # each dimension

  print('Read data')
  svr = svm.SVR(kernel='linear', verbose=1, max_iter=10000)
  model = svr.fit(data, np.ravel(labels))
  pickle.dump(model, open(os.path.join(cachedir, 'models/', 'svr_linear.pkl'), 'wb'))
  y = model.predict(data)
  print min(y), max(y)


def readData():
  cachepath = os.path.join(cachedir, 'data.npz')
  if os.path.exists(cachepath):
    mp = np.load(cachepath)
    return (mp['data'], mp['labels']) 
  else:
    data, labels = readDataFromDisk()
    np.savez_compressed(cachepath, data = data, labels = labels)
    return (data, labels)

def readDataFromDisk():
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  with open(trainlistpath) as f:
    testlist = [int(el) for el in f.read().splitlines()]
  allfeats = np.empty((0, 4096))
  allscores = np.empty((0, 1))
  for el in testlist:
    # el is 1 indexed, use it as el - 1
    imgpath = imgslist[el - 1][:-3] + 'txt'
    feats = np.loadtxt(os.path.join(featdir, imgpath))
    scores = np.loadtxt(os.path.join(scoresdir, str(el) + '.txt')).reshape(-1, 1)
    sel = np.random.choice(range(np.shape(feats)[0]), RAND_SEL)
    feats = feats[sel]
    scores = scores[sel]
    allfeats = np.concatenate((allfeats, feats), 0)
    allscores = np.concatenate((allscores, scores), 0)
    print('Read %d' % el)
  return (allfeats, allscores)

if __name__ == '__main__':
  main()

