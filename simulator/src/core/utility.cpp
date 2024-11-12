#include "utility.hpp"

#include <algorithm>
using std::shuffle;
#include <functional>
using std::multiplies;
#include <iomanip>
using std::setfill;
using std::setw;
#include <iostream>
using std::cout;
using std::endl;
#include <fstream>
using std::ifstream;
using std::ofstream;
#include <mutex>
using std::mutex;
using std::unique_lock;
#include <numeric>
using std::accumulate;
using std::iota;
#include <random>
using std::default_random_engine;
using std::random_device;
#include <sstream>
using std::stringstream;
#include <stdexcept>
using std::runtime_error;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <time.h>

#include <filesystem>
using std::filesystem::current_path;
using std::filesystem::directory_iterator;
using std::filesystem::exists;
using std::filesystem::is_directory;
using std::filesystem::is_regular_file;
using std::filesystem::path;
using std::filesystem::recursive_directory_iterator;

mutex mu_;

void PrintElapsedTime(struct timespec &start_time) {
  struct timespec end_time;
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  double seconds = (double)(end_time.tv_sec - start_time.tv_sec);
  double nanoseconds =
      ((double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000000);
  double elapsed_time = seconds + nanoseconds;

  cout << endl << "Elapsed time: " << elapsed_time << endl;
}

void PrintInput(string op, string in, int batch) {
  unique_lock<mutex> lk{mu_};

  if (batch == 0) {
    string input = "Input: " + in;
    printf("%-40s%-40s\n", op.c_str(), input.c_str());
  } else {
    // cout << "\r";
    cout << "Batch: " << batch << endl;
  }
}

void PrintCalibration(int layer) {
  unique_lock<mutex> lk{mu_};
  // cout << "\r";
  cout << "Entropy calibration for layer " << layer << endl;
}

void PrintBatchFlags(int batch, int num_batches) {
  cout << "\n";
  cout << "Collecting values at batch, total : " << num_batches
       << ", now : " << batch;
}

void GetImageFilePaths(vector<string> &input_file_paths, string dir) {
  path root = current_path();
  root /= dir;

  const vector<string> exts = {".jpg", ".jpeg", ".png", ".JPEG"};
  if (exists(root) && is_directory(root)) {
    for (auto const &entry : recursive_directory_iterator(root)) {
      for (auto const &ext : exts) {
        if (is_regular_file(entry) && entry.path().extension() == ext) {
          input_file_paths.push_back(entry.path().string());
        }
      }
    }
  }
}

void GetNumpyFilePaths(vector<string> &input_file_paths, string dir_) {
  const string ext{".npy"};
  path dir{dir_};

  //! FIXME: error propagation
  if (!exists(dir) || !is_directory(dir)) {
    return;
  }
  for (auto const &entry : recursive_directory_iterator(dir)) {
    if (is_regular_file(entry) && entry.path().extension() == ext) {
      input_file_paths.push_back(canonical(entry.path()).string());
    }
  }
}

void GetAbsoluteFilePath(string &file_path, string dir) {
  path root = current_path();
  root /= dir;
  file_path = root.string();
}

int *GetRandomNumbers(int start, int end) {
  int size = end - start;
  int *arr = new int[size];
  random_device rd;

  iota(arr, arr + size, start);
  default_random_engine rng(rd());
  shuffle(arr, arr + size, rng);

  return arr;
}

bool CheckFilePathReadable(const string &file_path) {
  path absolute_path = current_path() / file_path;
  ifstream file;
  file.open(absolute_path.string().c_str());
  if (file.fail()) {
    file.close();
    return false;
  } else {
    file.close();
    return true;
  }
}

bool CheckFilePathWritable(const string &file_path) {
  path absolute_path = current_path() / file_path;
  ofstream file;
  file.open(absolute_path.string().c_str());
  if (file.fail()) {
    file.close();
    return false;
  } else {
    file.close();
    return true;
  }
}

bool CheckDirectoryExist(const string &dir) {
  path total = current_path();
  total /= dir;

  if (!(exists(total) && is_directory(total))) {
    return false;
  }
  return true;
}

bool CheckDirectoryNumFiles(const string &dir, int num_file) {
  path total = current_path();
  total /= dir;

  int cnt = std::count_if(directory_iterator(total), directory_iterator(),
                          static_cast<bool (*)(const path &)>(is_regular_file));

  return cnt == num_file;
}