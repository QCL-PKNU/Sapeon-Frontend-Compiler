# Simulator

Simulator는 Sapeon NPU의 추론을 CPU 또는 GPU 상에서 시뮬레이션하는 프로그램입니다.

Sapeon NPU는 기존 TensorFlow, PyTorch, ONNX, darknet 등의 framework에서 사용하는 모델 형식들을 변환한 SpearGraph라는 모델 형식을 사용합니다.

Simulator는 이렇게 만들어진 SpearGraph가 잘 변환되었는지를 검증할 수 있는 기능을 지원합니다. 변환된 SpearGraph를 파싱하여 추론을 실행하고 그 결과를 layer 별로 파일로 출력합니다. 이 출력된 파일들을 변환 전 모델의 추론 결과와 비교해 모델이 제대로 동작하는지 검증할 수 있습니다.

단순히 형식이 변환된 모델만으로는 낮은 비트 수의 자료형으로 연산을 하는 Sapeon NPU에서는 사용이 불가능합니다. 그렇기 때문에 모델에 미리 학습된 높은 비트 수의 자료형(FP32)을 가진 값들을 낮은 비트 수의 자료형으로 변환하는 양자화 과정을 거쳐야 합니다.

이 양자화 과정을 Simulator에서는 Post-Training-Quantization 중 Static Quantization 기법으로 구현하고 있습니다. Sample Dataset을 사용한 Calibration을 통해 양자화할 값들의 범위를 결정하고 이 범위들로 양자화를 진행합니다. Simulator에서는 max, percentile, kl divergence 총 3가지 방법의 Calibration 기법을 제공합니다.

또한 Sapeon NPU에서의 연산을 CPU에서 구현한 시뮬레이션 기능을 제공합니다. 모델을 양자화해 준비해두었다면 CPU 상에서도 양자화된 값들을 사용하는 Sapeon NPU와 동일한 연산으로 추론해볼 수 있습니다.

# Simulator 모듈

## Network

Network는 모델의 신경망 전체에 해당하는 인스턴스입니다. SpearGraph 바이너리 파일을 읽고 Simulator에서 사용할 수 있도록 변환합니다. 신경망의 기본 단위인 Layer들을 들고 있습니다. Backend는 이 Network 인스턴스를 가지고 있고 각각의 Layer를 거치며 연산을 진행합니다.

## Layer

Layer는 신경망의 기본 단위입니다. 한 Layer는 텐서 연산을 수행할 수 있는 정보를 담고 있습니다. 예를 들어 Convolution Layer의 경우는 SpearGraph에서 읽어들인 Filter와 Bias, padding, stride 등의 텐서 연산에 필요한 parameter들을 가지고있습니다. 한 Layer에서는 1개 이상의 텐서 연산을 수행할 수 있습니다.

## Tensor

Tensor는 N차원 배열을 랩핑한 클래스입니다. Convolution의 Filter, Bias등과 같은 데이터 또한 Tensor 클래스로 랩핑되어서 Tensor 연산에 사용됩니다. Layer의 입력과 출력 Activation들 또한 Tensor 클래스로 랩핑되어서 Simulator 내부에서 사용합니다.

## Backend

Backend는 Simulator의 기능을 수행하는 주체입니다. Calibration, Quantization, Inference의 세 가지 기능을 조합하여 사용자가 수행하고 싶은 기능을 수행합니다. Backend에서 모델을 이루고 있는 Operation들을 연산할 프로세서를 선택할 수 있습니다. cpu 혹은 nvidia gpu를 사용할 수 있으며, 현재는 cpu에서만 Sapeon NPU 시뮬레이션 기능을 제공합니다.

## Calibration

Calibration 모듈에서는 Sample Dataset을 사용해 양자화할 데이터의 범위를 결정합니다. 각 Layer의 Output Activation Tensor, Layer이 들고있는 Filter, Bias 등의 Tensor, Network에 들어오는 이미지 Tensor의 양자화 데이터 범위를 결정합니다. 이 중 각 Layer의 Output Activation Tensor의 범위를 결정하는 방법으로 3가지 기법 중 하나를 선택할 수 있습니다.

1. Max Calibration : 전체 데이터 중 가장 큰 절댓값을 가지는 값을 기준으로 양자화 범위를 결정합니다.
2. Percentile Calibration : 전체 데이터를 히스토그램으로 만들어 절댓값이 가장 큰 값부터 제외합니다. 남아있는 값들의 수가 입력한 percentile 만큼이 될때를 기준으로 양자화 범위를 결정합니다.
3. KL Divergence Calibration : 전체 데이터를 히스토그램으로 만들고 모든 양자화 범위에 대해서 양자화된 분포를 생성합니다. 이렇게 만들어진 양자화된 분포와 기존 분포의 KL Divergence 값 중 최솟값에 해당하는 양자화 범위를 기준으로 양자화 범위를 결정합니다.
4. KL Divergence 2 Calibration : 기존 KL Divergence를 개선한 알고리즘입니다.

## Quantization

Quantization 모듈에서는 Calibration에서 계산된 양자화 범위를 사용해 Layer의 텐서 연산에 필요한 Tensor parameter들을 양자화하고 Sapeon NPU 연산에 필요한 데이터들을 초기화합니다.

## Inference

Inference 모듈에서는 추론을 실행합니다. FP 자료형으로 이루어진 신경망에 대한 추론을 실행함으로써 SpearGraph가 잘 변환되었는지를 확인할 수 있습니다. 양자화가 완료된 신경망의 경우는 Sapeon NPU 추론 과정을 시뮬레이션해 결과를 확인할 수 있습니다.

## Operation

Operation은 텐서 연산을 수행하는 연산 단위입니다. Simulator에서는 다양한 종류의 텐서 연산을 지원합니다. 각 Backend마다 사용 가능한 Operation의 종류는 다르며, FP 연산 지원 여부와 NPU 연산 시뮬레이션 여부를 다음 표에 표시해두었습니다.

- Supported CPU Backend Operations

| Operation | FP64 | FP32 | Sapeon NPU Simulation |
|--|--|--|--|
| Convolution | O | O | O |
| Connected | O | O | O |
| Maxpool | O | O | O |
| Avgpool | O | O | O |
| Softmax | O | O | X |
| Route | O | O | X |
| Reorg | O | O | X |
| Element-Wise Addition | O | O | O |
| Upsample | O | O | X |
| Group Convolution | O | O | O |
| Batch Normalization | O | O | X |
| Bias Addition | O | O | X |
| Pixelshuffle | O | O | X |
| Activation-Identity | O | O | O |
| Activation-ReLU | O | O | O |
| Activation-LeakyReLU | O | O | X |
| Activation-ReLU6 | O | O | X |
| Activation-Sigmoid | O | O | X |
| Activation-Mish | O | O | X |

- Supported FP Nvidia GPU Backend Operations

| Operation | FP64 | FP32 | Sapeon NPU Simulation |
|--|--|--|--|
| Convolution | O | O | X |
| Connected | O | O | X |
| Maxpool | O | O | X |
| Avgpool | O | O | X |
| Softmax | O | O | X |
| Route | O | O | X |
| Reorg | O | O | X |
| Element-Wise Addition | O | O | X |
| Upsample | O | O | X |
| Group Convolution | O | O | X |
| Batch Normalization | O | O | X |
| Bias Addition | O | O | X |
| Pixelshuffle | O | O | X |
| Add | O | O | X |
| ArgMax | O | O | X |
| ArgMin | O | O | X |
| Clip | O | O | X |
| Convolution Transpose | O | O | X |
| Gemm | O | O | X |
| Instance Normalization | O | O | X |
| Layer Normalization | O | O | X |
| MatMul | O | O | X |
| Mean | O | O | X |
| Mul | O | O | X |
| Reduce Sum | O | O | X |
| Resize | O | O | X |
| Sqrt | O | O | X |
| Sub | O | O | X |
| Sum | O | O | X |
| Activation-Identity | O | O | X |
| Activation-ReLU | O | O | X |
| Activation-LeakyReLU | O | O | X |
| Activation-ReLU6 | O | O | X |
| Activation-Sigmoid | O | O | X |
| Activation-CELU | O | O | X |
| Activation-ELU | O | O | X |
| Activation-Mish | O | O | X |
| Activation-PReLU | O | O | X |
| Activation-SELU | O | O | X |

## Prerequisites

- docker

[Install Docker Engine](https://docs.docker.com/engine/install/)

- nvidia-docker

[nvidia-docker](https://github.com/NVIDIA/nvidia-docker/)


## Getting Started

1. Git clone
```bash
git clone https://github.com/SAPEON-Artiference/aixgraph_simulator.git
```

2. Environment setup
```bash
cd aixgraph-simulator/docker
sudo sh docker_build.sh
./docker_run.sh
```

3. Compile Simulator
```bash
cd aixgraph-simulator

# First time running simulator
sudo ./install-mc.sh
./bootstrap.sh

# Build simulator
./build_simulator.sh
```

3.1 Python interface install

이 과정은 `build` 디렉토리에 `libspsim.so`가 있다는 것을 가정합니다.

```bash
python setup.py bdist_wheel
pip install dist/sapeon.simulator-*.whl
```

4. Run

- 커맨드라인 옵션 출력
```bash
./simulator --help
```

- Calibration 수행 후 모델 및 calibration table을 파일로 출력
```bash
./simulator --backend cpu \
            --model-path original-model.pb \
            --graph-type spear_graph \
            --dump-level default \
            --calib \
            --calibration-method percentile \
            --calibration-image-dir calib-images \
            --calibration-percentile 0.999 \
            --dump-calibrated-model \
            --calibrated-model-dump-path calibrated-model.pb \
            --dump-calibration-table \
            --calibration-table-dump-path calibration-table.txt
```

- Calibration 완료된 모델에 대해 x220 quantization 및 inference 수행
```bash
./simulator --backend cpu \
            --model-path calibrated-model.pb \
            --graph-type spear_graph \
            --dump-level debug \
            --quant \
            --quant-simulator x220 \
            --infer \
            --image-path dog.jpg
```

- 원본 모델에 대해 inference 수행
```bash
./simulator --backend cpu \
            --model-path original-model.pb \
            --graph-type spear_graph \
            --infer \
            --image-path dog.jpg
```

- Calibration, quantization, inference를 순차적으로 수행
```bash
./simulator --backend cpu \
            --model-path original-model.pb \
            --graph-type spear_graph \
            --preprocess-config-path config.json \
            --dump-level debug \
            --dump-dir test-results/ \
            --calib \
            --calibration-method entropy \
            --calibration-image-dir calib-images \
            --dump-calibrated-model \
            --calibrated-model-dump-path calibrated-model.pb \
            --dump-calibration-table \
            --calibration-table-dump-path calibration-table.txt
            --quant \
            --quant-simulator x220 \
            --infer \
            --image-path dog.jpg
```

- Collect 수행
```bash
./simulator --backend cpu \
            --model-path original-model.pb \
            --graph-type spear_graph \
            --preprocess-config-path config.json \
            --collect \
            --collect-image-dir calib-images \
            --collect-quant-max-dump-path dump/quant.max
```

- Collect 수행 후 X330 Quantization 및 Inference 수행
```bash
./simulator --backend cpu \
            --model-path original-model.pb \
            --graph-type spear_graph \
            --preprocess-config-path config.json \
            --collect \
            --collect-image-dir calib-images \
            --collect-quant-max-dump-path dump/quant.max \
            --quant \
            --quant-simulator x330 \
            --quant-cfg-path x330-config \
            --quant-max-path dump/quant.max \
            --infer \
            --image-path dog.jpg
```

- 원본 모델에 대해 validation 수행
```bash
./simulator --backend cpu \
            --model-path original-model.pb \
            --graph-type spear_graph \
            --preprocess-config-path config.json \
            --valid \
            --validation-image-dir images/imagenet_validation
```

- Compile & Run Tests
```bash
./prepare_test.sh  # if first time
./run_test.sh
```

- Lint Codes
```bash
./lint.sh
cat lint_result.include.txt
cat lint_result.src.txt
```

## Command-line options

### 공통

아래 옵션 필수로 요구

```bash
--backend [cpu | cudnn]
--model-path PATH
--graph-type [spear_graph | aix_graph]
```

아래 옵션부터는 선택 가능 옵션

### Preprocess

```bash
--preprocess-config-path PATH
# 지정 안할 경우 default preprocess 진행
```

Sample Preprocess Config
```json
{
  "resize_size": 232,
  "crop_size": 224,
  "normalization": {
    "channel": [
      0,
      1,
      2
    ],
    "scale": [
      1.0,
      1.0,
      1.0
    ],
    "gamma": [
      0.485,
      0.456,
      0.406
    ],
    "beta": [
      0.229,
      0.224,
      0.225
    ],
    "bias": [
      0.0,
      0.0,
      0.0
    ]
  }
}
```

Default Preprocess
- no resize
- no center crop
- scales = [1.0, 1.0, 1.0]
- gammas = [0.0, 0.0, 0.0]
- betas = [1.0, 1.0, 1.0]
- biases = [0.0, 0.0, 0.0]
- channels = [0, 1, 2]

preprocess 과정
1. resize_size 필드가 있을 경우 해당 값으로 resize 진행 (resize_size X resize_size)
2. CenterCrop 진행 (crop_size 필드 설정 여부에 따라 다르게 동작)
    - crop_size 필드가 있는 경우 해당 값으로 center crop 진행 (crop_size X crop_size) input size와 맞지 않으면 Warning 로그를 내보내고 input size로 center crop 진행
    - crop_size 필드가 있지 않은 경우 image의 width와 height 중 짧은 쪽의 값을 기준으로 center crop 후 input size에 맞춰 resize (darknet center crop 구현과 동일)
3. Normalization 진행 (scales, gammas, betas, biases, channels 사용)
    - 만약 normalization 필드가 없다면 default scales, gammas, betas, biases, channels 사용해 Normalization 진행

### Dump

```bash
--dump-level [none | default | debug]
# 지정 안할 경우 'default'
```

- `dump-level`
    - `none` : dump 파일 생성하지 않음
    - `default` : darknet-quant에서 `QUANT_DUMP_DEBUG=0` 인 경우에 대응
    - `debug` : darknet-quant에서 `QUANT_DUMP_DEBUG=1` 인 경우에 대응

`--dump-level` 옵션 값이 `none`이 아닐 시 아래 옵션 선택 가능

```bash
--dump-dir PATH
# 지정 안할 경우 'dump/'
```

- `dump-dir`
    - 유의: 아래의 `calibrated-model-dump-path` 등의 옵션과는 별개로 동작함

### Calibration

`--calib` 옵션 지정 시 수행

`--calib` 옵션 지정 시 아래 옵션 필수로 요구

```bash
--calibration-method [max | percentile | entropy]
--calibration-image-dir PATH
```

`--calibration-method=percentile` 인 경우 아래 옵션 필수로 요구

```bash
--calibration-percentile FLOAT (0 이상 1 이하)
```

`--calib` 옵션 지정 시 아래 옵션 선택 가능

```bash
--dump-calibrated-model
--dump-calibration-table
```

`--dump-calibrated-model` 옵션 지정 시 아래 옵션 선택 가능

```bash
--calibrated-model-dump-path PATH
# 지정 안할 경우 'dump/calib.pb'
```

`--dump-calibration-table` 옵션 지정 시 아래 옵션 선택 가능

```bash
--calibration-table-dump-path PATH
# 지정 안할 경우 'dump/calib-table.txt'
```

### Collect

`--collect` 옵션 지정 시 수행. 현재는 x330 시뮬레이션을 실행할 때 필요한 quant.max파일을 생성하기 위해 임시로 도입된 기능.

`--collect` 옵션 지정 시 아래 옵션 필수로 요구

```bash
--collect-image-dir PATH
```

`--collect-image-dir` 옵션은 quant.max 파일 생성에 필요한 이미지들이 존재하는 디렉토리를 지정하는 옵션. `--calibration-image-dir`과 동일한 역할.

`--collect` 옵션 지정 시 아래 옵션 선택 가능

```bash
--collect-quant-max-dump-path PATH
```

`--collect-quant-max-dump-path` 옵션은 quant.max 파일을 저장할 위치를 지정하는 옵션. 기본 경로는 `dump/quant.max`


### Quantization

`--quant` 옵션 지정 시 수행

`--quant` 옵션 지정 및 `--calib` 옵션 미지정 시 시뮬레이터는 `--model-path` 를 통해 입력받은 모델이 이미 calibration 수행이 완료된 상태로 가정. Calibration 수행이 되어 있지 않은 경우, 에러 메시지 출력 후 종료

`--quant` 옵션 지정 시 아래 옵션 필수로 요구

```bash
--quant-simulator [x220 | x330]
```

`--quant-simulator` 옵션이 x330일 경우 해당 옵션 지정 가능

```bash
--quant-cfg-path PATH
# 지정 안할 경우 default quant cfg가 적용됨.
--quant-max-path PATH
# 지정 안할 경우 fcalib mode를 설정해도 적용되지 않음. 
```

`--quant-simulator`옵션이 x330이고 `--quant-max-path` 옵션이 명시되었을 경우 해당 옵션 지정 가능

```bash
--quant-updated-ebias-dump-path PATH
# 지정 할 경우 updated.ebias가 저장될 경로를 지정할 수 있음.
```

### Inference

`--infer` 옵션 지정 시 수행

`--infer` 옵션 지정 시 아래 옵션 필수로 요구

```bash
--image-path PATH
```

### Validation

`--valid` 옵션 지정 시 수행

`--valid` 옵션 지정 시 아래 옵션 필수로 요구

```bash
--validation-image-dir PATH
```

`--validation-image-dir` 옵션은 validation을 진행할 이미지셋이 전체 포함된 디렉토리의 경로를 입력해야함. 해당 디렉토리는 class별 디렉토리로 나뉘어져 있고, 각 class 디렉토리에 해당 class에 속하는 이미지들이 포함됨.

### 시뮬레이터 동작 의사 코드

```python
## opt(x) : x는 command line option에 의해 결정되는 값

if not opt(calib) and not opt(quant) and not opt(infer):
    exit_with_msg('at least one of --calib, --quant, --infer, or --valid should be set')

if opt(calib):
    do calibration
    if opt(dump-calibrated-model):
        dump calibrated model
    if opt(dump-calibration-table):
        dump calibration table

if opt(collect):
    dump quant.max

if opt(quant):
    if model is not calibrated:
        do quantization with model default thresholds
    do quantization with calibrated thresholds

if opt(infer):
    if opt(quant):
        do input quantization
    do inference

if opt(valid):
    if opt(quant):
        do input quantization
    do validation
```

### 옵션 정리

```bash
--backend [cpu | cudnn]
--model-path PATH
--graph-type [spear_graph | aix_graph]
--preprocess-config-path PATH
--dump-level [none | default | debug]
--dump-dir PATH

--calib
--calibration-method [max | percentile | entropy]
--calibration-image-dir PATH
--calibration-percentile FLOAT
--dump-calibrated-model
--dump-calibration-table
--calibrated-model-dump-path PATH
--calibration-table-dump-path PATH

--collect
--collect-quant-max-dump-path PATH
--collect-image-dir PATH

--quant
--quant-simulator [x220 | x330]
--quant-cfg-path PATH
--quant-max-path PATH
--quant-updated-ebias-dump-path PATH

--infer
--image-path PATH

--valid
--validation-image-dir PATH
```

---

### 변경 사항 (v1.0.0 기준)

- 공통: 옵션명에 포함된 모든 underscore (`_`)를 hyphen (`-`)으로 변경

```bash
# 유지
--help, -h
--backend, -b [cpu | cudnn]
--preprocess-config-path, -f
--calibration-percentile, -P

# 이름만 변경
--protobuf, -p -> --model-path
--image-path, -i -> --calibration-image-dir
--calibration, -c -> --calibration-method

# 기능 변경
--should-save-each-layer-outputs, -S -> --dump-level

# 추가
--dump-dir PATH
--calib
--dump-calibrated-model
--dump-calibration-table
--calibrated-model-dump-path PATH
--calibration-table-dump-path PATH
--collect
--collect-quant-max-dump-path PATH
--collect-image-dir PATH
--quant
--quant-simulator [x220 | x330]
--quant-cfg-path PATH
--quant-max-path PATH
--quant-updated-ebias-dump-path PATH
--infer
--image-path PATH
--valid
--validation-image-dir PATH

# 삭제
--datatype, -d
--image-extension, -I
--quantization, -q
--should-quantize-each-layers
--save, -s
--num-batches, -n
```
