# YOLO11 vs RT-DETRv2 — Reproducible COCO + Real‑Time Benchmark

[![Platform](https://img.shields.io/badge/Platform-Linux-informational.svg)](#requirements)
[![Benchmark](https://img.shields.io/badge/Type-Benchmark-blueviolet.svg)](#)
[![Reproducible](https://img.shields.io/badge/Reproducible-yes-success.svg)](#reproducibility)



A reproducible benchmark to compare **pretrained** object detectors **without fine‑tuning**:

- **YOLO11** — one‑stage detectors optimized for speed
- **RT‑DETRv2** — real‑time DETR‑style transformer detectors

It reports both:
- **Accuracy (offline):** COCO val2017 via COCOeval  
  `mAP@[0.50:0.95]`, `AP50`, `AP75`, `AP small/medium/large`
- **Performance (online‑like):** video throughput + tail latency  
  `FPS mean`, end‑to‑end `p50/p95/p99`

> In real‑time systems, **p95/p99 latency** often matters as much as (or more than) mAP.

---

## Table of contents
- [Features](#features)
- [Supported models](#supported-models)
- [Quickstart](#quickstart)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Project structure](#project-structure)

---

## Features
- **Reproducible** runs with fixed inference settings (`imgsz`, `conf`, `iou`)
- **Sequential multi‑model** execution with CSV outputs
- Saves **COCO‑format predictions** (`runs/preds/*.json`) for auditability
- Robust error handling (e.g., **OOM**) — benchmark continues
- Data download script: **COCO val2017** + **annotations** + sample **OpenCV video**

---

## Supported models

### YOLO11
- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`

### RT‑DETRv2
- `PekingU/rtdetr_v2_r18vd`
- `PekingU/rtdetr_v2_r34vd`
- `PekingU/rtdetr_v2_r50vd`
- `PekingU/rtdetr_v2_r101vd`

---

## Quickstart

### 1) Clone and enter
```bash
git clone https://github.com/Haldorsfjall/realtime-object-detection-benchmark.git>
cd realtime-object-detection-benchmark
```

### 2) Create venv + install deps (CUDA PyTorch)
```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

### 3) Download COCO val2017 + annotations + sample video
```bash
bash scripts/download_data.sh
```

### 4) Run the full benchmark (smoke + COCO + video)
```bash
python scripts/runner.py --device cuda --fp16 \
  --models yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt \
           PekingU/rtdetr_v2_r18vd PekingU/rtdetr_v2_r34vd PekingU/rtdetr_v2_r50vd PekingU/rtdetr_v2_r101vd
```

### 5) CPU mode (for quick testing)
```bash
python scripts/runner.py --device cpu \
  --models yolo11n.pt PekingU/rtdetr_v2_r18vd
```

---

## Outputs

All artifacts are written to `runs/`:

- `runs/check_build.csv` — smoke step (load models + tiny inference)
- `runs/coco_metrics.csv` — COCOeval metrics per model
- `runs/video_bench.csv` — FPS and end‑to‑end latency percentiles per model
- `runs/preds/*.json` — COCO‑format predictions
- `runs/manifest.txt` — environment manifest (library versions, GPU/driver, etc.)

Tip: to merge COCO + video into one table, use pandas:
```bash
python -c "import pandas as pd; \
c=pd.read_csv('runs/coco_metrics.csv'); \
v=pd.read_csv('runs/video_bench.csv'); \
print(c.merge(v, on=['model','status'], how='outer').sort_values('mAP@[.5:.95]', ascending=False).head(20))"
```

---

## Configuration

### Inference defaults
- `imgsz=640`
- `conf=0.001` (low threshold, closer to PR curve evaluation)
- `iou=0.7`

### Adjust runtime settings
- Skip stages:
  - `--skip-smoke`
  - `--skip-coco`
  - `--skip-video`
- Video parameters (runner):
  - `--warmup` (default 50)
  - `--frames` (default 500)

---

## Project structure
```text
scripts/
  runner.py            # runs smoke + COCO + video sequentially
  check_build.py       # quick load/inference sanity test
  eval_coco.py         # COCO val2017 eval + COCOeval metrics
  bench_video.py       # FPS + latency percentiles on a video
  download_data.sh     # downloads COCO val2017 + annotations + sample video
  utils/               # loaders, COCO helpers, IO, error handling
data/                  # (gitignored) datasets and video
runs/                  # (gitignored) results and predictions
```

---

# (RU) YOLO11 vs RT‑DETRv2 — воспроизводимый бенчмарк COCO + real‑time

Воспроизводимый бенчмарк для сравнения **предобученных** детекторов объектов **без дообучения**:

- **YOLO11** — одношаговые детекторы, оптимизированные под скорость
- **RT‑DETRv2** — real‑time DETR‑подобные детекторы на трансформерах

Проект измеряет два класса метрик:
- **Качество (offline):** COCO val2017 через COCOeval  
  `mAP@[0.50:0.95]`, `AP50`, `AP75`, `AP small/medium/large`
- **Производительность (приближенно online):** пропускная способность и хвосты задержек  
  `FPS mean`, end‑to‑end `p50/p95/p99`

> В реальных real‑time системах **p95/p99** часто важны так же, как (или важнее) mAP.

---

## Содержание
- [Возможности](#features)
- [Поддерживаемые модели](#supported-models)
- [Быстрый старт](#quickstart)
- [Файлы результатов](#outputs)
- [Настройка](#configuration)
- [Структура проекта](#project-structure)

---

## Возможности
- **Воспроизводимые** прогоны с фиксированными настройками инференса (`imgsz`, `conf`, `iou`)
- **Последовательный запуск** нескольких моделей с выводом результатов в CSV
- Сохранение **предсказаний в COCO‑формате** (`runs/preds/*.json`) для проверки/аудита
- Устойчивая обработка ошибок (например, **OOM**) — бенчмарк не падает целиком
- Скрипт скачивания данных: **COCO val2017** + **аннотации** + видео‑пример **OpenCV**

---

## Поддерживаемые модели

### YOLO11
- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`

### RT‑DETRv2
- `PekingU/rtdetr_v2_r18vd`
- `PekingU/rtdetr_v2_r34vd`
- `PekingU/rtdetr_v2_r50vd`
- `PekingU/rtdetr_v2_r101vd`

---

## Быстрый старт

### 1) Клонировать и перейти в папку
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_FOLDER>
```

### 2) Создать venv и установить зависимости (CUDA PyTorch)
```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

### 3) Скачать COCO val2017 + аннотации + видео‑пример
```bash
bash scripts/download_data.sh
```

### 4) Запустить полный бенчмарк (smoke + COCO + video)
```bash
python scripts/runner.py --device cuda --fp16 \
  --models yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt \
           PekingU/rtdetr_v2_r18vd PekingU/rtdetr_v2_r34vd PekingU/rtdetr_v2_r50vd PekingU/rtdetr_v2_r101vd
```

### 5) CPU‑режим (для быстрой проверки)
```bash
python scripts/runner.py --device cpu \
  --models yolo11n.pt PekingU/rtdetr_v2_r18vd
```

---

## Файлы результатов

Все артефакты сохраняются в `runs/`:

- `runs/check_build.csv` — этап smoke (загрузка моделей + короткий тест инференса)
- `runs/coco_metrics.csv` — метрики COCOeval по каждой модели
- `runs/video_bench.csv` — FPS и квантили end‑to‑end задержек по видео
- `runs/preds/*.json` — предсказания в COCO‑формате
- `runs/manifest.txt` — манифест окружения (версии библиотек, GPU/драйвер и т.д.)

Подсказка: объединить COCO + video в одну таблицу через pandas:
```bash
python -c "import pandas as pd; \
c=pd.read_csv('runs/coco_metrics.csv'); \
v=pd.read_csv('runs/video_bench.csv'); \
print(c.merge(v, on=['model','status'], how='outer').sort_values('mAP@[.5:.95]', ascending=False).head(20))"
```

---

## Настройка

### Настройки инференса по умолчанию
- `imgsz=640`
- `conf=0.001` (низкий порог — ближе к оценке по PR‑кривой)
- `iou=0.7`

### Управление этапами
- Пропустить этапы:
  - `--skip-smoke`
  - `--skip-coco`
  - `--skip-video`
- Параметры видео (runner):
  - `--warmup` (по умолчанию 50)
  - `--frames` (по умолчанию 500)

---

## Структура проекта
```text
scripts/
  runner.py            # последовательно запускает smoke + COCO + video
  check_build.py       # smoke: загрузка + мини-инференс
  eval_coco.py         # COCO val2017 + COCOeval
  bench_video.py       # FPS + квантили задержек на видео
  download_data.sh     # скачивает COCO val2017 + аннотации + видео
  utils/               # загрузчики моделей, COCO утилиты, IO, обработка ошибок
data/                  # (gitignore) датасеты и видео
runs/                  # (gitignore) результаты и предсказания
```
