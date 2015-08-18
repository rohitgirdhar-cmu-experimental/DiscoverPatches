
# @input hits: a array of size(retrival_corpus) with True if it's a hit
# and false otherwise
# @input poscount: Number of elements in retrieval corpus of target class
# Code completely inspired for Oxford Buildings evaluation code
def computeAP(hits, poscount):
  old_recall = 0.0
  old_precision = 1.0
  ap = 0.0

  intersect_size = 0.0
  i = 0
  j = 0
  while i < len(hits):
    if hits[i]:
      intersect_size += 1
    recall = intersect_size / poscount
    precision = intersect_size / (j + 1.0)
    ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

    old_recall = recall
    old_precision = precision
    j += 1
    i += 1

  return ap
