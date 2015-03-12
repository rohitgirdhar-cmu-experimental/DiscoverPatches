from sklearn import preprocessing, svm, linear_model
import numpy as np
import os, pickle
import locker
import pdb

featdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/features/CNN_fc7_text'
imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
querylistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TestList.txt'
cachedir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch'
modelfname = 'svr_linear_10000.pkl'
resultsdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/all_query_scores/' + os.path.splitext(modelfname)[0] + '/'
if not os.path.exists(resultsdir):
  os.makedirs(resultsdir)

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
    feats = np.loadtxt(os.path.join(featdir, imgslist[el - 1])[:-3] + 'txt') # IMP, 1 indx
    y = predict(feats, scaler, model)
    np.savetxt(outpath, y, fmt = '%0.10f', delimiter = '\n')
    locker.unlock(outpath)
    print('Done for %d' % el)

def readModels():
  modelcache = os.path.join(cachedir, 'models/', modelfname)
  if os.path.exists(modelcache):
    model = pickle.load(open(modelcache, 'rb'))
  else:
    print ('Unable to load the model from %s' % modelcache)
    return None
  scalercache = os.path.join(cachedir, 'std_scaler.pkl')
  if os.path.exists(scalercache):
    scaler = pickle.load(open(scalercache, 'rb'))
  else:
    print ('Unable to load the std scaler')
  return (model, scaler)
  
def predict(data, scaler, model):
  data = preprocessing.normalize(data, norm='l2')
  data = scaler.transform(data)
  y = model.predict(data)
  return y

if __name__ == '__main__':
  main()

