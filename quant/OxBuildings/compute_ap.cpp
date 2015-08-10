#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <sstream>

using namespace std;

vector<string>
load_list(const string& fname)
{
  vector<string> ret;
  ifstream fobj(fname.c_str());
  if (!fobj.good()) { cerr << "File " << fname << " not found!\n"; exit(-1); }
  string line;
  while (getline(fobj, line)) {
    ret.push_back(line);
  }
  return ret;
}

template<class T>
set<T> vector_to_set(const vector<T>& vec)
{ return set<T>(vec.begin(), vec.end()); }

float
compute_ap(const set<string>& pos, const set<string>& amb, const vector<string>& ranked_list)
{
  float old_recall = 0.0;
  float old_precision = 1.0;
  float ap = 0.0;
  
  size_t intersect_size = 0;
  size_t i = 0;
  size_t j = 0;
  for ( ; i<ranked_list.size(); ++i) {
    if (amb.count(ranked_list[i])) continue;
    if (pos.count(ranked_list[i])) intersect_size++;

    float recall = intersect_size / (float)pos.size();
    float precision = intersect_size / (j + 1.0);

    ap += (recall - old_recall)*((old_precision + precision)/2.0);

    old_recall = recall;
    old_precision = precision;
    j++;
  }
  return ap;
}

float
run_compute_ap(string gtq, const vector<string>& ranked_list, bool ignore_junk_images = true)
{
  set<string> good_set = vector_to_set( load_list(gtq + "_good.txt") );
  set<string> ok_set = vector_to_set( load_list(gtq + "_ok.txt") );
  set<string> junk_set = vector_to_set( load_list(gtq + "_junk.txt") );
  if (!ignore_junk_images) {
    junk_set.clear();
  }

  set<string> pos_set;
  pos_set.insert(good_set.begin(), good_set.end());
  pos_set.insert(ok_set.begin(), ok_set.end());

  float ap = compute_ap(pos_set, junk_set, ranked_list);
  
  return ap;
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

string removeExt(string a) {
  // remove the .jpg
  return a.substr(0, a.size() - 4);
}

void readOutputs(string fpath, const vector<string>& imgslist, map<string, vector<string>>& outputs) {
  ifstream fin(fpath);
  string line;
  while (getline(fin, line)) {
    vector<string> temp = split(line, ' ');
    vector<string> matches;
    for (int i = 1; i < temp.size(); i++) {
      vector<string> match = split(temp[i], ':');
      if (stoi(match[0]) == -1) {
        // this is the mysterious extra image in OxBuildings
        continue;
      }
      matches.push_back(removeExt(imgslist[stoi(match[0]) - 1]));
    }
    outputs[removeExt(imgslist[stoi(temp[0]) / 10000 - 1])] = matches;
  }
  fin.close();
}

int main(int argc, char* argv[]) {
  string gtdir = string(argv[1]);
  string qlistfpath = string(argv[2]);
  string outfile = string(argv[3]);
  vector<string> imgslist = load_list(argv[4]);
  bool ignore_junk_images = true;
  if (argc > 5) {
    if (argv[5][0] == '1') {
      cout << "NOt Ignoring junk" << endl;
      ignore_junk_images = false;
    }
  }

  map<string, vector<string>> outputs;
  readOutputs(outfile, imgslist, outputs);
  ifstream fin(qlistfpath.c_str());
  string qpath, temp;
  int qid, n = 0;
  float map = 0;
  while (fin >> qpath >> temp >> qid) {
    string gtq = gtdir + "/" + qpath;
    float ap = run_compute_ap(gtq, outputs[temp], ignore_junk_images);
    map += ap;
    n += 1;
  }
  cout << "mAP : " << map / n << endl;
  fin.close();
  return 0;
}

