---
icon: note
tags: [guide]
order: 60
---

# Moreh 솔루션 업데이트 하기 (`update-moreh`)

Moreh 솔루션은 주기적으로 업데이트되면서 솔루션의 전반적 성능이 개선되고 있습니다. Moreh 솔루션을 활용하는 방식에 따라 특정 버전의 Moreh 솔루션만을 사용하실 수 있지만, 가급적 최신 Moreh 솔루션을 사용하시는 것을 권장하고 있습니다. Moreh 솔루션을 업데이트하시면 사용하시는 환경의 Deep learning framework(PyTorch, TensorFlow) 및 Moreh driver 등의 필수 패키지들이 업데이트됩니다.


Moreh 솔루션은 다음 명령어를 통해 업데이트하실 수 있습니다.

```bash
$ update-moreh
```

기본적으로 위 명령어 실행 시 현재까지 배포된 버전 중 최신 버전으로 업데이트를 진행합니다.

- `-target`

Moreh 솔루션을 특정 버전으로 다운(업)그레이드를 할 수 있는 옵션입니다. `--target` 옵션 뒤에는 특정 버전을 아래와 같이 기입해주시면 됩니다.

```bash
# 예전 버전인 22.7.2 버전으로 다운그레이드하는 예시
update-moreh --target 22.7.2
```

```bash
# 23.9.0 버전으로 업데이트 및 롤백하는 예시
update-moreh --target 23.9.0
```

## Deep Learning Framework 버전 변경하기

Moreh 솔루션은 Pytorch 1.7.1 버전뿐만이 아닌 Pytorch 1.10.0, 1.13.1 버전과 Tensorflow 2.9.0 버전에 대해서도 제공하고 있습니다.

다른 버전의 Framework 설치를 위한 옵션은 다음과 같습니다.

### PyTorch 버전 변경하기

```bash
# Pytorch 1.10.0 버전 설치 torch 기본 버전은 1.7.1
update-moreh --torch 1.10.0

# 특정 Moreh 솔루션 버전으로 Pytorch 1.10.0버전 설치
update-moreh --torch 1.10.0 --target 23.9.0
```

### TensorFlow 버전 변경하기

```bash
# tensorflow 2.9.0 (기본 버전) 설치
update-moreh --tensorflow
# 특정 Moreh 솔루션 버전으로 TensorFlow 설치
update-moreh --tensorflow --target 23.9.0
```

💡 Tensorflow와 Pytorch 1.10.0 혹은 1.13.1 버전은 **동시에 설치를 할 수 없습니다**.

```bash
ubuntu@vm:~$ update-moreh --torch 1.10.0 --tensorflow
update-moreh does not support tensorflow and torch>=1.10.0 at the same time
Try update-moreh --tensorflow or update-moreh --torch
```