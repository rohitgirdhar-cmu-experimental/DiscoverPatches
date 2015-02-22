from sklearn import preprocessing, svm, linear_model
import numpy as np
import os, pickle
import locker
import pdb

featdir = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/features/CNN_fc7_text'
imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/ImgsList.txt'
querylistpath = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/split/QueryList.txt'
resultsdir = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/learn_good_patches/scratch/query_scores'
cachedir = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/learn_good_patches/scratch'

def main():
  model, scaler = readModels()
  # for every test image, read in all features, predict and store
  with open(querylistpath) as f:
    querylist = [int(el) for el in f.read().splitlines()]
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  for el in querylist:
    outpath = os.path.join(resultsdir, str(el) + '.txt')
    if not locker.lock(outpath):
      continue
    feats = np.loadtxt(os.path.join(featdir, imgslist[el])[:-3] + 'txt')
    y = predict(feats, scaler, model)
    np.savetxt(outpath, y, fmt = '%0.10f', delimiter = '\n')
    locker.unlock(outpath)
    print('Done for %d' % el)

def readModels():
  modelcache = os.path.join(cachedir, 'models/', 'svr_rbf.pkl')
  if os.path.exists(modelcache):
    model = pickle.load(open(modelcache, 'rb'))
  else:
    print ('Unable to load the model')
  scalercache = os.path.join(cachedir, 'std_scaler.pkl')
  if os.path.exists(scalercache):
    scaler = pickle.load(open(scalercache, 'rb'))
  else:
    print ('Unable to load the std scaler')
  return (model, scaler)
  
def predict(data, scaler, model):
  data = scaler.transform(data)
  y = model.predict(data)
  return y

if __name__ == '__main__':
  main()

