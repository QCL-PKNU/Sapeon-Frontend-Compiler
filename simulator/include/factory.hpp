#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>

template <class Base>
class Factory {
  using create = std::unique_ptr<Base> (*)();

 public:
  static auto &GetFactoryMap() {
    static std::map<std::string, create> map;

    return map;
  }

  static bool RegisterCreateFunction(std::string name, create function) {
    // std::cout << "RegisterCreateFunction(" << name << ")" << std::endl;
    GetFactoryMap()[name] = function;

    return true;
  }

  static std::unique_ptr<Base> CreateInstance(std::string name) {
    // std::cout << "CreateInstance(" << name << ")" << std::endl;
    std::unique_ptr<Base> p_instance = nullptr;
    auto it = GetFactoryMap().find(name);

    if (it != GetFactoryMap().end()) p_instance = GetFactoryMap().at(name)();

    return p_instance;
  }
};

template <class Base>
class CalibrationFactory {
  using create = std::unique_ptr<Base> (*)(int, std::optional<float>);

 public:
  static auto &GetFactoryMap() {
    static std::map<std::string, create> map;

    return map;
  }

  static bool RegisterCreateFunction(std::string name, create function) {
    // std::cout << "RegisterCreateFunction(" << name << ")" << std::endl;
    GetFactoryMap()[name] = function;

    return true;
  }

  static std::unique_ptr<Base> CreateInstance(std::string name, int num_layers,
                                              std::optional<float> percentile) {
    // std::cout << "CreateInstance(" << name << ")" << std::endl;
    std::unique_ptr<Base> p_calibration = nullptr;
    auto it = GetFactoryMap().find(name);
    if (it != GetFactoryMap().end()) {
      p_calibration = GetFactoryMap().at(name)(num_layers, percentile);
    }
    return p_calibration;
  }
};

#endif  // FACTORY_HPP
