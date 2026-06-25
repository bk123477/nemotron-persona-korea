# Nemotron-Personas-Korea Parquet Helpers

이 디렉터리에는 `Nemotron-Personas-Korea` 데이터셋의 `parquet` 파일을 더 쉽게 사용할 수 있도록 변환하는 도구가 들어 있습니다.

## 목적
- 기존 `data` 폴더는 그대로 유지
- 로컬 `parquet` 파일이 Git LFS 포인터로 남아 있어도, Hugging Face 원격 데이터를 자동으로 사용
- RAG / LangChain / LangGraph / 온톨로지 구축에 적합한 `JSONL` 포맷으로 변환

## 설치
가상환경이 활성화된 상태에서 다음을 실행하세요:

```bash
cd /home/minkih/nemotron-persona-korea
.venv/bin/pip install datasets pyarrow pandas
```

## 사용법
### 샘플 변환
처음에는 1,000개 샘플만 생성해서 동작을 확인하세요:

```bash
cd /home/minkih/nemotron-persona-korea
.venv/bin/python3 scripts/convert_personas_jsonl.py --sample 1000
```

### 전체 변환
모든 데이터를 변환하려면 `--sample` 옵션 없이 실행합니다:

```bash
.venv/bin/python3 scripts/convert_personas_jsonl.py
```

### Hugging Face 원격 데이터 사용
로컬 `parquet` 파일이 없거나 Git LFS로 인해 접근 불가능할 때 자동으로 `nvidia/Nemotron-Personas-Korea` 원격 데이터셋을 사용합니다.

필요한 경우 다른 Hugging Face 리포지토리를 지정할 수도 있습니다:

```bash
.venv/bin/python3 scripts/convert_personas_jsonl.py --hf-repo nvidia/Nemotron-Personas-Korea --hf-split train
```

## 출력 파일
기본 출력 경로는 다음과 같습니다:

- `Nemotron-Personas-Korea/data/jsonl/personas.jsonl`

각 줄은 다음 형태의 JSON 객체입니다:

```json
{
  "id": "...",
  "text": "...",
  "metadata": {
    "uuid": "...",
    "sex": "...",
    "age": 30,
    ...
  }
}
```

## 추가 활용
- `text` 필드는 RAG/검색 인덱싱에 적합한 텍스트 문서
- `metadata` 필드는 LangChain/온톨로지 구축 시 속성 기반 필터링에 사용
