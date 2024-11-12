#include "calibration/calibrator.hpp"

#include <string>
using std::string;

namespace spgraph_simulator {
namespace calibration {
void Calibrator::CalibrationResult::AsTextfile(
    const string &path_to_out_textfile) {
  FILE *fp = fopen(path_to_out_textfile.c_str(), "w");

  const auto &[_, in_thr] = this->ranges.at(0);
  fprintf(fp, "input_tensor:0\t%8.6f\n", in_thr);

  for (int i = 1; i < this->ranges.size(); ++i) {
    const auto &[name, thr] = this->ranges.at(i);
    fprintf(fp, "%s\t%8.6f\n", name.c_str(), thr);
  }
  fclose(fp);
}
}  // namespace calibration
}  // namespace spgraph_simulator
