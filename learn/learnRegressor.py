from sklearn import preprocessing, svm, linear_model
import numpy as np
import os, pickle
import h5py
import pdb

#featdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/features/CNN_fc7_text'
DATA_CACHE_FILE = 'data_cache/data_pool5.h5'
FEAT_DIM = 9216
OUTDIR = 'models/pool5/'
featdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/features/CNN_pool5_text'
imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
trainlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TrainList_120.txt'
scoresdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_scores'
cachedir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch'
RAND_SEL = 800 # randomly select these many from each image
kernelname = 'rbf'
niter = 10000
modelcache = os.path.join(cachedir, OUTDIR, 'svr_' + kernelname + '_' + str(niter) + '_pool5.pkl')

def main():
  ## read the data
  data, labels = readData()
  origData = data
  origLabels = labels
  print('Read data')
  print modelcache
  if os.path.exists(modelcache):
    model = pickle.load(open(modelcache, 'rb'))
  else:
  #  data = data[np.arange(2000)]
  #  labels = labels[np.arange(2000)]
    data = preprocessing.normalize(data, norm='l2')
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data) # each dimension
    pickle.dump(scaler, open(os.path.join(cachedir, 'std_scaler.pkl'), 'wb'))
    print ('Scaled the data')

    svr = svm.SVR(kernel=kernelname, verbose=1, max_iter=niter)
    #svr = svm.SVR(kernel='poly', verbose=1, max_iter=10000)
    #lr = linear_model.LinearRegression()
    #lr = linear_model.Ridge(alpha = 0.5)
    #model = svr.fit(data, np.ravel(labels))
    #model = svr.fit(data, np.ravel(labels), n_jobs=12)
    model = svr.fit(data, np.ravel(labels))
    pickle.dump(model, open(modelcache, 'wb'))
  y = model.predict(origData)
  testset = np.random.choice(np.arange(np.shape(labels)[0]), 10000)
  pdb.set_trace()
  print model.score(origData[testset], origLabels[testset])

def readData():
  cachepath = os.path.join(cachedir, DATA_CACHE_FILE)
  if os.path.exists(cachepath):
    h5f = h5py.File(cachepath, 'r')
    data = h5f['data'][:]
    labels = h5f['labels'][:]
    return (data, labels)
  else:
    data, labels = readDataFromDisk()
    h5f = h5py.File(cachepath, 'w')
    h5f.create_dataset('data', data=data)
    h5f.create_dataset('labels', data=labels)
    h5f.close()
    return (data, labels)

def readDataFromDisk():
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  with open(trainlistpath) as f:
    testlist = [int(el) for el in f.read().splitlines()]
  allfeats = np.empty((0, FEAT_DIM))
  allscores = np.empty((0, 1))
  for el in testlist:
    # el is 1 indexed, use it as el - 1
    imgpath = imgslist[el - 1][:-3] + 'txt'
    feats = np.loadtxt(os.path.join(featdir, imgpath))
    scores = np.loadtxt(os.path.join(scoresdir, str(el) + '.txt')).reshape(-1, 1)
    nsel = min(RAND_SEL, np.shape(feats)[0])
    sel = np.random.choice(range(np.shape(feats)[0]), nsel)
    feats = feats[sel]
    scores = scores[sel]
    allfeats = np.concatenate((allfeats, feats), 0)
    allscores = np.concatenate((allscores, scores), 0)
    print('Read %d' % el)
  return (allfeats, allscores)

if __name__ == '__main__':
  main()

