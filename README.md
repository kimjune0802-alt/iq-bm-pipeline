# I/Q bm Pipeline

국가근로 계측 데이터 업무에서 I/Q 로그와 가중치 파일을 자동으로 읽어 복소수 연산, bm 테이블 생성 및 최종 4라인 검증값 산출까지 처리하는 Python 파이프라인입니다.  
반복적인 bm 생성·합산 작업을 완전히 자동화하기 위해 구현했습니다.

## Features

- `log_mult_ch.csv`, `log_mult_wgt.csv` 자동 파싱
- 복소수 I/Q × 가중치 복소수 연산
- bm 테이블 블록(48라인 단위) 자동 인식 및 합산
- 48라인 합계를 4라인으로 폴딩(fold)한 최종 검증값 산출
- Windows 배치 파일을 통한 원클릭 실행 지원

## Usage

```bash
python log_mult_pipeline_v6.py --ch log_mult_ch.csv --wgt log_mult_wgt.csv --outdir .

bat
_parsing.bat
