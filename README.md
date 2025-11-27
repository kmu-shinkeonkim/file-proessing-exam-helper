# File Processing Exam Helper - Tree Visualizer

이진 탐색 트리(BST), AVL 트리, B-Tree 구현체를 자동으로 시각화하는 도구입니다.

> 주의사항 
> - 이 도구는 C++로 작성된 트리 구현체의 **빌드된 실행 파일**을 필요로 합니다.
> - `.cpp` 소스 파일이 아닌, 컴파일된 **바이너리 실행 파일**을 지정해야 합니다.
> - 따라서, 트리 구현체를 제대로 작성하지 않았거나 컴파일하지 않은 경우, 이 도구는 정상적으로 작동하지 않습니다.
> - Python3로 실행됩니다.

## 목차

- [주요 기능](#주요-기능)
- [사용 방법](#사용-방법)
- [빠른 시작](#빠른-시작)
- [설정 가이드](#설정-가이드)

## 주요 기능

### 1. 구현체 실행
- BST, AVL Tree, B-Tree 구현체 자동 실행
- 랜덤 또는 커스텀 테스트 케이스 생성
- Insert Only / Insert + Delete 두 가지 시나리오 테스트

### 2. 시각화
- CLI 트리 그래픽
- 이진 트리와 B-Tree 지원
- 단계별 또는 최종 결과만 선택 가능

### 3. 인터랙티브 모드
- 각 단계마다 Enter 키로 진행
- 학습, 디버깅, 발표에 최적화

## 사용 방법

### 1. 저장소 복제

```bash
# 또는 Git을 사용하여 전체 저장소 복제
git clone https://github.com/kmu-shinkeonkim/file-proessing-exam-helper.git
```

### 2. File-Processing-Report-Tester C++ 파일 빌드

**중요**: 실행 파일은 C++ 소스를 컴파일한 **바이너리 파일**이어야 합니다.

```bash
# BST 빌드 예시
g++ -o submit/BST submit/학번_이름_BST.cpp

# AVL 빌드 예시
g++ -o submit/AVL submit/학번_이름_AVL.cpp

# B-Tree 빌드 예시
g++ -o submit/BT submit/학번_이름_BT.cpp

# 실행 권한 부여 (Linux/Mac)
chmod +x submit/BST
chmod +x submit/AVL
chmod +x submit/BT
```

### 3. 빌드된 실행 파일을 `submit/` 디렉토리에 배치

- `submit/BST` (또는 `submit/BST.exe` on Windows)
- `submit/AVL` (또는 `submit/AVL.exe` on Windows)
- `submit/BT` (또는 `submit/BT.exe` on Windows)

### 4. 요구사항

- Python 3.6 이상
- 빌드된 C++ 실행 파일 (`.cpp`가 아닌 실행 가능한 바이너리)

---

## 빠른 시작

### 1. 실행 파일 경로 설정

`tree_visualizer.py` 파일을 열고 **30-35번째 줄**의 실행 파일 경로를 수정하세요:

```python
# ===== Executable Paths =====
# submit 디렉토리 기준 상대 경로 (확장자 없는 빌드된 실행 파일)
BST_EXECUTABLE = "BST"      # 본인의 파일명으로 변경
AVL_EXECUTABLE = "AVL"      # 본인의 파일명으로 변경
BTREE_EXECUTABLE = "BT"     # 본인의 파일명으로 변경
```

⚠️ **주의사항**:
- `.cpp` 확장자가 아닌 **빌드된 실행 파일명**을 입력하세요
- 예: `"BST.cpp"` ❌ → `"BST"` ✅
- 테스트하지 않을 트리는 `None`으로 설정하세요

### 2. 실행

```bash
python tree_visualizer.py
```

### 3. 출력 예시

```
================================================================================
CONFIGURATION
================================================================================
SKIP_PROGRESS:        True
  → Show only final results
SHOW_ALL_IN_ONE_TIME: True
  → Display all at once

Test case: Random generation
           15 insertions, 8 deletions

Executables:
  BST:    김신건_20191564_BST
  AVL:    김신건_20191564_AVL
  B-Tree: 김신건_20191564_BT
================================================================================

Testing BST: 김신건_20191564_BST

Test 1: Insert Operations Only

Final tree structure (raw): <<< 3 > 5 < 7 >> 10 < 15 >>

Visualization:
    10
  ┌────┐
  │    │
  5   15
┌──┐
│  │
3  7
```

---

## 설정 가이드

### 디스플레이 설정

파일 **19-28번째 줄**에서 설정:

```python
# ===== Display Settings =====
SKIP_PROGRESS = True            # True: 최종 결과만 / False: 모든 단계 표시
SHOW_ALL_IN_ONE_TIME = True     # True: 한번에 출력 / False: 단계별 Enter 입력
```

#### `SKIP_PROGRESS`

| 값 | 동작 | 용도 |
|---|------|------|
| `True` | 각 테스트의 최종 결과만 표시 | 빠른 결과 확인 |
| `False` | 모든 insert/delete 단계마다 트리 상태 표시 | 학습, 디버깅 |

#### `SHOW_ALL_IN_ONE_TIME`

| 값 | 동작 | 용도 |
|---|------|------|
| `True` | 모든 결과를 한번에 출력 | 스크롤하며 확인, 스크린샷 |
| `False` | 각 섹션마다 Enter 키 입력 필요 | 발표, 시연 |

### 실행 파일 설정

파일 **30-35번째 줄**에서 설정:

```python
# ===== Executable Paths =====
BST_EXECUTABLE = "BST"      # BST 실행 파일명 (빌드된 바이너리)
AVL_EXECUTABLE = "AVL"      # AVL 실행 파일명 (빌드된 바이너리)
BTREE_EXECUTABLE = "BT"     # B-Tree 실행 파일명 (빌드된 바이너리)
```

**예시 1**: 모든 트리 테스트
```python
BST_EXECUTABLE = "BST"
AVL_EXECUTABLE = "AVL"
BTREE_EXECUTABLE = "BT"
```

**예시 2**: BST만 테스트
```python
BST_EXECUTABLE = "BST"
AVL_EXECUTABLE = None          # 건너뛰기
BTREE_EXECUTABLE = None        # 건너뛰기
```

### 테스트 케이스 설정

파일 **37-50번째 줄**에서 설정:

```python
# ===== Test Case Settings =====
CUSTOM_INSERT_VALUES = None     # None: 랜덤 / [10, 5, 15, ...]: 직접 지정
CUSTOM_DELETE_VALUES = None     # None: 랜덤 / [5, 15, ...]: 직접 지정

# 랜덤 생성 시 크기
RANDOM_CREATE_COUNT = 15
RANDOM_DELETE_COUNT = 8
```

#### 모드 1: 랜덤 생성 (기본)

```python
CUSTOM_INSERT_VALUES = None
CUSTOM_DELETE_VALUES = None
RANDOM_CREATE_COUNT = 15        # 15개 랜덤 삽입
RANDOM_DELETE_COUNT = 8         # 8개 랜덤 삭제
```

#### 모드 2: 완전 커스텀

```python
CUSTOM_INSERT_VALUES = [10, 5, 15, 3, 7, 12, 20]
CUSTOM_DELETE_VALUES = [5, 15]
```

#### 모드 3: 하이브리드

```python
CUSTOM_INSERT_VALUES = [10, 5, 15, 3, 7, 12, 20]
CUSTOM_DELETE_VALUES = None  # 위 값들 중에서 랜덤 선택
RANDOM_DELETE_COUNT = 3  # 3개 삭제
```

## 🤝 기여

버그 리포트나 기능 제안은 [Issues](https://github.com/kmu-shinkeonkim/file-proessing-exam-helper/issues)에 등록해주세요.
