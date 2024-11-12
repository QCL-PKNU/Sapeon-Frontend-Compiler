# PySpear Project

PySpear Project는 Spear Graph를 쉽게 편집 및 생성하기 위한 프레임워크입니다.
코드를 통해 모델의 스펙을 정의하고, export하는 것으로 간단히 Protobuf Serialized Spear Graph 를 생성할 수 있습니다.

## Install
### Requirements
Python >= 3.8 (virtual environment 추천)
```bash
$ pwd
/path/to/aixgraph_simulator/pyspear
```

### Bootstrap
```bash
$ ./bootstrap.sh
```

### Run Example Code
```bash
$ ./run_main.sh
```

### Testing
```bash
$ ./run_test.sh
```

## Add New Operation
1. 새로운 Operation은 `pyspear/nn` 디렉토리에 추가합니다.

2. Operation은 `pyspear/nn/node.py`의 Node 클래스를 상속받아 작성합니다. 필요한 작성 전에 `pyspear/nn/node_util.py` 등 `*_util.py` 파일들을 확인하고 작업합니다.

3. 이후 Operation을 변환하기 위해 converter를 작성합니다. converter는 `pyspear/converter/<graph-version>` 디렉토리에 추가합니다.

4. converter는 `pyspear/converter/node_converter.py`의 NodeConverter를 상속받아 작성합니다. 이 또한 `pyspear/converter/<graph-version>/*_util.py` 파일을 참고하여 작성합니다.

5. Graph에 converter를 추가합니다. `pyspear/graph_<graph-version>.py` 파일의 `self.converters`에 방금 구현한 converter를 추가합니다.

참고사항
- import를 쉽게 하려면 `__init__.py`에 추가하면 좋습니다.
- activation은 `pyspear/nn/activation.py`에 구현합니다. (activation을 protobuf class로 변환하는 코드는 `pyspear/converter/<graph-version>/converter_utils_<graph-version>.py`를 참고합니다.)


## Add New Graph
기본적으로 serialize를 자유롭게 구현할 수 있습니다. 꼭 converter들을 사용하지 않아도 됩니다.


1. 새로운 Graph는 `pyspear/graph` 디렉토리에 추가합니다.

2. `pyspear/graph/graph.py`의 Graph를 상속받아 작성합니다.

3. 미리 구현해둔 `pyspear/converter/<graph-version>`들을 node를 변환시킬 `self.converters`에 추가합니다.
