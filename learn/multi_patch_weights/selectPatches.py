import numpy as np
# takes unary scores, pairwise sims and param to compute selection
# Both inputs are np.array
# noMatchesExist is list of boxes for this image for which no results exits
# hence should not be considered (no way to fix it yet) (0-indexed)
# output is 0 indexed
def selectPatches(unary, sims, param1, nsel, noMatchesExist):
  # follow a greedy strategy to select
  assert(np.size(unary) == np.shape(sims)[0] == np.shape(sims)[1])
  # sims is already normalized to between 0 and 1
  # normalize the unary to between 0 and 1
#  unary = unary - min(unary)
#  unary = unary / (max(unary) + 0.001)
  toppatches = np.argsort(-np.array(unary))
  nPatches = np.shape
  sel = []
  sel_scores = []
  for i in range(nsel):
    # iterate through all the elements
    top_perf_score = -1e6
    top_perf = -1
    for j in range(np.size(unary)):
      if j in noMatchesExist:
        continue
      if j in sel:
        continue
      # compute the score for this patch
      max_sim = -1
      for selected in sel:
        max_sim = max(max_sim, sims[j][selected])
      thisscore = computeScore(unary[j], max_sim, param1)
      if thisscore > top_perf_score:
        top_perf_score = thisscore
        top_perf = j
    sel.append(top_perf)
    sel_scores.append(top_perf_score)
  return (sel,sel_scores)

def computeScore(un, maxsim, param):
  return un + param * maxsim

