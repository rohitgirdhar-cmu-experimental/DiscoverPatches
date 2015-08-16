#!/usr/bin/python -tt

# computes top matches for every image, using a given a scoring

import os, sys, math, subprocess, random
import numpy as np
sys.path.append('../')
from computeScores_DCG import computeDCG
from nms import non_max_suppression_fast
sys.path.append('../learn/multi_patch_weights/')
from selectPatches import selectPatches
import h5py
import random

takeTopN = 1 # -n = random n patches
              # -1 = 1 random patch
              # 1 = top match
              # 5 = top 5 matches
param1 = -0
upto = 1 # 0=> select nth. 1=> select 1..nth (only valid for top matches, not random)
nmsTh = 0.9 # set = -1 for no NMS
          # else, set a threshold between [0, 1]
use_similarity_selection = False # = True for using the similarity scores for multi patch sel
N_OUTPUT = 99999999;
NMATCHES_PER_PATCH = 999999;

if 0:
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_refined/'
  selboxdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TestList.txt'
#  method = 'svr_rbf_10000'
#  method = 'gt'
  method = 'svr_linear_FullData_liblinear'
#  method = 'svr_linear_FullData_liblinear_pool5'
#  method = 'deep_regressor_5K'
  scoresdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/all_query_scores/' + method
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/retrievals/' + method + '__' + str(takeTopN) + '__nms' + str(nmsTh) + '.txt'
elif 0:
  # for full img matching case
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/fullImg/'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTest.txt'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/fullImg.txt'
  simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0010_ExtendedPAL_moreTest/matches/CNN/fullImg/'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/NdxesTest.txt'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/fullImg.txt'
#  simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for patch case
  method = 'patch'
  use_similarity_selection = True
  upto = 1
  takeTopN = 5
  param1 = -0.2
  if takeTopN > 1:
    NMATCHES_PER_PATCH = 50;
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/test/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/test_final_patch.txt'
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/query_scores/fc7_PeopleOnly/'
  simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 1:
  # for patch case
  method = 'patch'
  use_similarity_selection = False
  upto = 1
  takeTopN = 1
  param1 = -0.2
  if takeTopN > 1:
    NMATCHES_PER_PATCH = 50;
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0010_ExtendedPAL_moreTest/matches_refined/CNN/test/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/NdxesPeopleTest.txt'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0010_ExtendedPAL_moreTest/matches_top/test_final_patch.txt'
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/query_scores/CNN/test/'
#  simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/aux_matches/matches_fullImg/matches_refined/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TestList.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/aux_matches/matches_fullImg/matches_top.txt'
elif 0:
  # Hussian,for patch case
  method = 'patch'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0007_HussianHotels/matches_refined/test/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0007_HussianHotels/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0007_HussianHotels/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0007_HussianHotels/matches_top/test.txt'
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/query_scores/fc7_TrainOnly/'
  simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches_bin/' # just dummy here
  nmsTh = -1 # set = -1 for no NMS
  param1 = 0
elif 0:
  # Hussian,for full img matching case
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0007_HussianHotels/matches_refined/fullImg/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0007_HussianHotels/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0007_HussianHotels/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0007_HussianHotels/matches_top/fullImg.txt'
  simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case (BOW)
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/fullImg_bow+gv/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/fullImg_bow+gv.txt'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for patch case
  FULL_MATCH_WT = 9 # this x the score for full image
  
  use_similarity_selection = True
  upto = 1
  takeTopN = 5
  param1 = -0.2
  if takeTopN > 1:
    NMATCHES_PER_PATCH = 50;

  method = 'patch+full'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/test.txt'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/test/'
  fullmatchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/fullImg/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/test_final_deep.txt'
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/query_scores/fc7_PeopleOnly/'
  simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/Jegou13/001_basic/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/Jegou13.txt'
  simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case (Jegou - with hes aff features)
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/Jegou13_hesaff/'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTest.txt'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/Jegou13_hesaff.txt'
#  simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case (Jegou - with hes aff features)
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/Jegou13_hesaff_heatmap_0.6/'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTest.txt'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/Jegou13_hesaff_heatmap.txt'
  simsmatdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case (Jegou - with hes aff features - DAVID baselines)
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/david_baselines/matches_HMAPsaliency_0.2/'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTest.txt'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/Jegou13_hesaff_heatmap_davidBaselines_saliency.txt'
  simsmatdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case (Jegou - with hes aff features - DAVID baselines)
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/david_baselines/matches_HMAPfaceHeatmap2/'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTest.txt'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/Jegou13_hesaff_heatmap_faceHeatmap2.txt'
  simsmatdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for the grid search
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/train_GridSearch/mean/'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTrain_noNeg.txt'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/Jegou13_hesaff_heatmap_gridsearch.txt'
  simsmatdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for the grid search
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/train_GridSearch/david_FaceHeatmap2/0.9/'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTrain_noNeg.txt'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/Jegou13_hesaff_heatmap_gridsearch.txt'
  simsmatdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/Jegou13/001_basic/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/Jegou13.txt'
  simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # OxBuildings,for patch case
  method = 'patch'
  get_class_style = 'oxford'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/All.txt'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_refined/test/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_top/test.txt'
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/query_scores/fc7/'
  #simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/learn/pairwise_matches_bin/' # just dummy here
  nmsTh = -1 # set = -1 for no NMS
  param1 = -0.2
elif 0:
  # for full img matching case
  method = 'full-img'
  get_class_style = 'oxford'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/All.txt'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_refined/fullImg'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_top/fullImg.txt'
  #simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case

  use_similarity_selection = False
  upto = 1
  takeTopN = 5
  param1 = -0.0
  if takeTopN > 1:
    NMATCHES_PER_PATCH = 50;

  FULL_MATCH_WT = 9 # this x the score for full image
  get_class_style = 'oxford'
  method = 'patch+full'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/All.txt'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_refined/test/'
  fullmatchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_refined/fullImg/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_top/testfullpatch.txt'
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/query_scores/fc7/'
  simsmatdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/learn/pairwise_matches/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case
  method = 'full-img'
  get_class_style = 'oxford'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_refined/Jegou13_hesaff_heatmap'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_top/Jegou13_hesaff_heatmap.txt'
  #simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case
  method = 'full-img'
  get_class_style = 'oxford'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_refined/Jegou13_hesaff'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_top/Jegou13_hesaff.txt'
  #simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for full img matching case (Jegou - with hes aff features)
  method = 'full-img'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_withResize/matches_heatmap_0.5'
  retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTest.txt'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/Jegou13_hesaff_heatmap.txt'
  simsmatdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches/'
  nmsTh = -1 # set = -1 for no NMS


MAXBOXPERIMG = 10000

def main():
  # read images list
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  with open(testlistpath) as f:
    testlist = [int(t) for t in f.read().splitlines()]
  with open(retrievallistpath) as f:
    retlist = [int(t) for t in f.read().splitlines()]

  fout = open(outfpath, 'w')
  allscores = np.zeros((1, 8))
  allscores_cls = {} # same as allscores, except separately for each cls
  numdists_cls = {}
  for i in testlist:
    print i
    try:
      with open(os.path.join(scoresdir, str(i) + '.txt')) as f:
        patchscores = [float(el) for el in f.read().splitlines()]
    except:
      patchscores = [1] # assuming only 1 patch in the image and with score = 1
    order = np.argsort(-np.array(patchscores)) # to reverse sort
  
    if nmsTh >= 0:
      order = performNMS(order, os.path.join(selboxdir, str(i) + '.txt'), nmsTh)

    if not use_similarity_selection:
      selected = []
      if takeTopN < 0:
        selected = random.sample(range(len(order)), -takeTopN)
      elif takeTopN > 0 and upto == 1:
        selected = list(order[:takeTopN])
      elif takeTopN > 0 and upto == 0:
        selected = [order[takeTopN - 1]]
    else:
      if not (method == 'full-img'):
        sims = np.zeros((0,0))
        try:
          sims = readHDF5(os.path.join(simsmatdir_bin, str(i) + '.h5'), 'sims')
        except:
          print 'Unable to read bin sims file. Trying txt'
        if np.shape(sims)[0] == 0:
          try:
            sims = np.loadtxt(os.path.join(simsmatdir, str(i) + '.txt'))
          except:
            print 'Unalbe to read bin as txt. Using 0s'
            sims = np.zeros((len(patchscores), len(patchscores)))

        selected,_ = selectPatches(np.array(patchscores), sims, param1, takeTopN, [])
        if upto == 0:
          selected = [selected[takeTopN - 1]]
      else:
        selected = [0]

    # get the top matches from each and intersection
    if method == 'patch+full':
      matches = readMatchesWithFull(matchesdir, fullmatchesdir, i, selected, FULL_MATCH_WT, retlist)
    else:
      matches = readMatches(matchesdir, i, selected, retlist)
    
    matches = randomSortZeroScores(matches)

    scores = computeScores(matches, i, imgslist)
    allscores += np.array(scores)
    # also add these scores to class specific lists as well
    testcls = getClass(i - 1, imgslist)
    if testcls in allscores_cls.keys():
      allscores_cls[testcls][0] += np.array(scores)
      allscores_cls[testcls][1] += 1
    else:
      allscores_cls[testcls] = [np.array(scores), 1]
    
    ndist = countMatchesOfClass(matches, imgslist, 10, 'NaturalImages')
    if testcls in numdists_cls.keys():
      numdists_cls[testcls][0] += ndist
      numdists_cls[testcls][1] += 1
    else:
      numdists_cls[testcls] = [ndist, 1]
    
    qboxes = [(i) * MAXBOXPERIMG + el + 1 for el in selected]
    fout.write('%s; ' % ','.join([str(el) for el in qboxes])) # query box
    for match in matches[:min(N_OUTPUT, len(matches))]:
      fout.write('%d:%f:%s ' % (match[1], match[0], 
            ','.join([str(el) for el in match[2]])))
    fout.write('\n')
  fout.close()
  print 'mp1,mp3,mp5,mp10,mp20,atleast1/3,atleast1/10,DCG/10'
  print ','.join([str(s) for s in list(allscores / len(testlist))[0]])
  # print for each class
  classes = allscores_cls.keys()
  print ','.join(classes)
  dcg_scores = [allscores_cls[cls][0][7] / allscores_cls[cls][1] for cls in classes]
  print ','.join([str(el) for el in dcg_scores])
  print ','.join([str(numdists_cls[cls][0] * 1.0 / numdists_cls[cls][1]) 
      for cls in classes])

# outputs [(score, imid, imfeatids)...] // imid is not the imid*10K+featid
def readMatches(matchesdir, i, boxids, retlist):
  fpath = os.path.join(matchesdir, str(i) + '.txt')
  lines = readLines(fpath, boxids)
  allmatches = []
  for line in lines:
    matches = []
    line_matches =  line.strip().split()
    for el in line_matches[:min(NMATCHES_PER_PATCH, len(line_matches))]:
      el2 = el.split(':')
      matches.append((float(el2[1]), int(el2[0])))
    allmatches.append(matches)
  matches = mergeRanklists(allmatches, retlist)
  return matches

# outputs [(score, imid, imfeatids)...] // imid is not the imid*10K+featid
def readMatchesWithFull(matchesdir, fullmatchesdir, i, boxids, FULL_MATCH_WT, retlist):
  # patch matches
  fpath = os.path.join(matchesdir, str(i) + '.txt')
  lines = readLines(fpath, boxids)
  allmatches = []
  for line in lines:
    matches = []
    line_matches =  line.strip().split()
    for el in line_matches[:min(NMATCHES_PER_PATCH, len(line_matches))]:
      el2 = el.split(':')
      matches.append((float(el2[1]), int(el2[0])))
    allmatches.append(matches)
  # full matches
  fpath = os.path.join(fullmatchesdir, str(i) + '.txt')
  lines = readLines(fpath, [0])
  for line in lines:
    matches = []
    line_matches =  line.strip().split()
    for el in line_matches[:min(NMATCHES_PER_PATCH, len(line_matches))]:
      el2 = el.split(':')
      matches.append((float(el2[1]) * FULL_MATCH_WT, int(el2[0])))
    allmatches.append(matches)

  matches = mergeRanklists(allmatches, retlist)
  return matches

# returns [(score, imgid, imfeatids)..]
# retlist is the list of ids all images in the corpus from which to retrieve
def mergeRanklists(allmatches, retlist):
  imid2score = {}
  imid2feats = {} # store what bounding boxes in this image matched

  # initialize the lists
  for retel in retlist:
    imid2score[retel] = 0
    imid2feats[retel] = []

  for matches in allmatches:
    for match in matches:
      imid = getImgId(match[1])
      #if imid not in imid2score.keys():
      #  imid2score[imid] = match[0]
      #  imid2feats[imid] = [match[1]]
      #else:
      if imid not in imid2score.keys():
        print ('err: ' + str(imid))
        continue
      imid2score[imid] += match[0]
      imid2feats[imid].append(match[1])
  res = imid2score.items()
  res = sorted(res, key=lambda tup: tup[1], reverse=True) # remember, reverse sort!
  res = [(m[1], m[0], imid2feats[m[0]]) for m in res]
  return res

# matches must be [(score, imid)...]
def computeScores(matches, imgid, imgslist):
  # remove the exact match
  matches2 = matches[:]
  sameornot = [m[1] == imgid for m in matches2]
  if sum(sameornot) > 0:
    del matches2[np.where(sameornot)[0][0]]

  clses = [getClass(m[1] - 1, imgslist) for m in matches2]
  cls = getClass(imgid - 1, imgslist)
  hits = [c == cls for c in clses]
  scores = []
  for i in [1,3,5,10,20]: # for mP
    scores.append(float(sum(hits[:i])) / i)
  scores.append(sum(hits[:3]) > 0) # for atleast 1/3
  scores.append(sum(hits[:10]) > 0)
  scores.append(computeDCG(hits[:10], 10))
  return scores

# matches must be [(score, imid)...]
def countMatchesOfClass(matches, imgslist, n, cls):
  clses = [getClass(m[1] - 1, imgslist) for m in matches][:n]
  hits = [c == cls for c in clses]
  return sum(hits)

def readLines(fpath, lnos): # lnos must be 0 indexed
  order = np.argsort(np.array(lnos))
  lines = []
  with open(fpath) as f:
    for i, line in enumerate(f):
      if i in lnos:
        lines.append(line)
  return list(np.array(lines)[order])

def getClass(imid, imgslist): 
  # imgid here must be 0-indexed!!!
  try:
    if get_class_style == 'oxford':
      return '_'.join(imgslist[imid].split('_')[:-1])
    else:
      return os.path.dirname(imgslist[imid])
  except NameError: # get_class_style not defined
    return os.path.dirname(imgslist[imid])

def getImgId(idx):
  return idx / MAXBOXPERIMG

def performNMS(order, selboxfpath, th):
  f = open(selboxfpath)
  boxes = [[float(el) for el in line.split(',')] 
      for line in f.read().splitlines()]
  f.close()
  # this order is descending, so store reverse of this order for nms
  # and then reverse back the output
  boxes = [(box[1],box[0],box[3],box[2]) for box in boxes]
  boxes_rev = []
  order_rev = []
  for i in order[::-1]:
    boxes_rev.append(boxes[i])
    order_rev.append(i)
  _, pick = non_max_suppression_fast(np.array(boxes_rev), th)
  order_rev = np.array(order_rev)
  return order_rev[pick]

def readHDF5(fpath, dbname):
  print fpath
  f = h5py.File(fpath, 'r')
  data = f[dbname][:]
  f.close()
  return data

def randomSortZeroScores(matches):
  random.seed(1)
  actual_matches = [el for el in matches if el[0] > 0]
  other_matches = [el for el in matches if el[0] == 0]
  random.shuffle(other_matches)
  actual_matches += other_matches
  return actual_matches

if __name__ == '__main__':
  main()

