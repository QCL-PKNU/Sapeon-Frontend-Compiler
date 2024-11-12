#include <iostream>
using std::string;
using std::cout;
using std::cerr;

#include "argparse.hpp"
#include "glog/logging.h"
#include "tl/expected.hpp"
using tl::expected;
using tl::make_unexpected;

#include "arguments.hpp"
#include "enums/error.hpp"
#include "backends/backend.hpp"

expected<void, SimulatorError> ParseArguments(int, char **, Arguments &);
expected<void, SimulatorError> Simulate(Arguments);


int main(int argc, char **argv) {
    cout << "###########################\n";
    cout << "######### Simulation Stated\n";
    cout << "###########################\n";

    google::SetLogDestination(google::GLOG_INFO, "./output_log");
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;

    Arguments args;

    auto result = ParseArguments(argc, argv, args);
    if (!result) {
        LOG(ERROR) << "Argument parsing failed";
        cerr << "Usage: simulator --help" << '\n';
        return 1;
    }

    result = Simulate(args);
    if (!result) {
        LOG(ERROR) << "Simulation failed";
        return 1;
    }

    return 0;
}


expected<void, SimulatorError> ParseArguments(int argc, char **argv,
                                              Arguments &args) {
                                                
    args.input_type = Arguments::InputType::kImage;

    argparse::ArgumentParser program("simulator", "1.0.0");

    // required
    program.add_argument("--backend", "-b")
        .required()
        .help("set simulation backend (cpu | cudnn)");
    program.add_argument("--model-path", "-p")
        .required()
        .help("set model binary path");
    program.add_argument("--graph-type", "-g")
        .required()
        .help("set graph type (spear_graph | aix_graph)");

    // optional
    program.add_argument("--preprocess-config-path", "-f")
        .help("set config file path");
    program.add_argument("--dump-level")
        .default_value(string("default"))
        .help("set dump level (none | default | debug)");
    program.add_argument("--dump-dir")
        .default_value(string("dump"))
        .help("set output dump directory");

    // calibration
    program.add_argument("--calib")
        .default_value(false)
        .implicit_value(true)
        .help("run calibration");
    program.add_argument("--calibration-method", "-c")
        .help("set calibration method (max | percentile | entropy | entropy2)");
    program.add_argument("--calibration-image-dir", "-i")
        .help("set calibration image directory path");
    program.add_argument("--calibration-batch-size", "-n")
        .default_value(1)
        .scan<'d', int>()
        .help(
            "set the number of images in each batch for calibration (default is "
            "1)");
    program.add_argument("--calibration-percentile", "-P")
        .scan<'g', float>()
        .help("set percentile calibration percentile");
    program.add_argument("--dump-calibrated-model")
        .default_value(false)
        .implicit_value(true)
        .help("dump calibrated model as file");
    program.add_argument("--dump-calibration-table")
        .default_value(false)
        .implicit_value(true)
        .help("dump calibration table as human readable format");
    program.add_argument("--calibrated-model-dump-path")
        .default_value(string("dump/calib.pb"))
        .help("set calibrated model dump path");
    program.add_argument("--calibration-table-dump-path")
        .default_value(string("dump/calib-table.txt"))
        .help("set calibration table dump path");

    // collect
    program.add_argument("--collect")
        .default_value(false)
        .implicit_value(true)
        .help("run collect. create quant.max for x330 quantization");
    program.add_argument("--collect-quant-max-dump-path")
        .default_value(string("dump/quant.max"))
        .help("set quant.max dump path");
    program.add_argument("--collect-image-dir")
        .help("set collect image directory path for quant.max");

    // quantization
    program.add_argument("--quant")
        .default_value(false)
        .implicit_value(true)
        .help("run quantization");
    program.add_argument("--quant-simulator")
        .help("set quantization simulator (x220 | x330)");
    program.add_argument("--quant-cfg-path")
        .help("set config file path when use x330 quantization");
    program.add_argument("--quant-max-path")
        .help("set max file path when use x330 filter quantization");
    program.add_argument("--quant-updated-ebias-dump-path")
        .default_value(string("dump/updated.ebias"))
        .help("set updated.ebias dump path");

    // inference
    program.add_argument("--infer")
        .default_value(false)
        .implicit_value(true)
        .help("run inference");
    program.add_argument("--image-path", "-I")
        .help("set image path for inference");

    // validation
    program.add_argument("--valid")
        .default_value(false)
        .implicit_value(true)
        .help("run validation");
    program.add_argument("--validation-image-dir")
        .help("set validation image directory path");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        LOG(ERROR) << err.what();
        return make_unexpected(SimulatorError::kArgumentsParsingError);
    }

    args.do_calib(program.get<bool>("--calib"));
    args.do_collect(program.get<bool>("--collect"));
    args.do_quant(program.get<bool>("--quant"));
    args.do_infer(program.get<bool>("--infer"));
    args.do_valid(program.get<bool>("--valid"));

    args.backend(program.get<string>("--backend"));
    args.graph_type(program.get<string>("--graph-type"));
    args.model_path(program.get<string>("--model-path"));

    if (program.is_used("--preprocess-config-path")) {
        args.preprocess_config_path(
            program.get<string>("--preprocess-config-path"));
    } else {
        args.preprocess_config_path(std::nullopt);
    }

    args.dump_level(program.get<string>("--dump-level"));

    if (args.dump_level() != "none") {
        args.dump_dir(program.get<string>("--dump-dir"));
    } else {
        args.dump_dir(std::nullopt);
    }

    if (program.is_used("--calibration-method")) {
        args.calibration_method(program.get<string>("--calibration-method"));
    } else {
        args.calibration_method(std::nullopt);
    }
    if (program.is_used("--calibration-batch-size")) {
        args.calibration_batch_size(
            static_cast<size_t>(program.get<int>("--calibration-batch-size")));
    } else {
        args.calibration_batch_size(std::nullopt);
    }
    if (program.is_used("--calibration-image-dir")) {
        args.calibration_image_dir(program.get<string>("--calibration-image-dir"));
    } else {
        args.calibration_image_dir(std::nullopt);
    }
    if (program.is_used("--calibration-percentile")) {
        args.calibration_percentile(program.get<float>("--calibration-percentile"));
    } else {
        args.calibration_percentile(std::nullopt);
    }

    args.dump_calibrated_model(program.get<bool>("--dump-calibrated-model"));
    args.dump_calibration_table(program.get<bool>("--dump-calibration-table"));

    if (program.is_used("--dump-calibrated-model")) {
        args.calibrated_model_dump_path(
            program.get<string>("--calibrated-model-dump-path"));
    } else {
        args.calibrated_model_dump_path(std::nullopt);
    }

    if (program.is_used("--dump-calibration-table")) {
        args.calibration_table_dump_path(
            program.get<string>("--calibration-table-dump-path"));
    } else {
        args.calibration_table_dump_path(std::nullopt);
    }

    if (program.is_used("--collect-quant-max-dump-path")) {
        args.collect_quant_max_path(
            program.get<std::string>("--collect-quant-max-dump-path"));
    } else {
        args.collect_quant_max_path(std::nullopt);
    }

    if (program.is_used("--collect-image-dir")) {
        args.collect_image_dir(program.get<std::string>("--collect-image-dir"));
    } else {
        args.collect_image_dir(std::nullopt);
    }

    if (program.is_used("--quant-simulator")) {
        args.quant_simulator(program.get<string>("--quant-simulator"));
    } else {
        args.quant_simulator(std::nullopt);
    }

    if (program.is_used("--quant-cfg-path")) {
        args.quant_cfg_path(program.get<string>("--quant-cfg-path"));
    } else {
        args.quant_cfg_path(std::nullopt);
    }

    if (program.is_used("--quant-max-path")) {
        args.quant_max_path(program.get<string>("--quant-max-path"));
    } else {
        args.quant_max_path(std::nullopt);
    }

    if (program.is_used("--quant-updated-ebias-dump-path")) {
        args.quant_updated_ebias_dump_path(
            program.get<string>("--quant-updated-ebias-dump-path"));
    } else {
        args.quant_updated_ebias_dump_path(std::nullopt);
    }

    if (program.is_used("--image-path")) {
        args.image_path(program.get<string>("--image-path"));
    } else {
        args.image_path(std::nullopt);
    }

    if (program.is_used("--validation-image-dir")) {
        args.validation_image_dir(program.get<string>("--validation-image-dir"));
    } else {
        args.validation_image_dir(std::nullopt);
    }

    return args.CheckArguments();
}

expected<void, SimulatorError> Simulate(Arguments args) {
    auto p_backend = Factory<Backend>::CreateInstance(args.backend());
    if (p_backend == nullptr) {
        const string msg = "Failed to create backend: " + args.backend();
        LOG(ERROR) << msg;
        return make_unexpected(SimulatorError::kCreateInstanceError);
    }

    auto result = p_backend.get()->Run(args);
    if (!result) return make_unexpected(result.error());
    return {};
}
