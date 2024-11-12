#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "network/tensor.hpp"

void PrintElapsedTime(struct timespec &start);

void PrintInput(std::string op, std::string in, int batch);

void PrintCalibration(int layer);

void PrintBatchFlags(int batch, int num_batches);

void GetImageFilePaths(std::vector<std::string> &input_file_paths,
                       std::string dir);

void GetNumpyFilePaths(std::vector<std::string> &input_file_paths,
                       std::string dir);

void GetAbsoluteFilePath(std::string &file_path, std::string dir);

bool CheckFilePathReadable(const std::string &path);

bool CheckFilePathWritable(const std::string &path);

bool CheckDirectoryExist(const std::string &dir);

bool CheckDirectoryNumFiles(const std::string &dir, int num_file);

int *GetRandomNumbers(int start, int end);

#endif  // UTILITY_HPP
