import os

def lock(fpath):
  lockpath = fpath + '.lock'
  if not os.path.exists(fpath) and not os.path.exists(lockpath):
    os.makedirs(lockpath)
    return True
  return False

def unlock(fpath):
  lockpath = fpath + '.lock'
  if os.path.exists(lockpath):
    os.rmdir(lockpath)
    return True
  return False

