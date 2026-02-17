#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
COCO_DIR="${DATA_DIR}/coco"
COCO_IMG_DIR="${COCO_DIR}/val2017"
COCO_ANN_DIR="${COCO_DIR}/annotations"

VAL_ZIP="${COCO_DIR}/val2017.zip"
ANN_ZIP="${COCO_DIR}/annotations_trainval2017.zip"

VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VTEST_URL="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
VTEST_PATH="${DATA_DIR}/vtest.avi"

say() { echo -e "\033[1;32m[download]\033[0m $*"; }

need_cmd() { command -v "$1" >/dev/null 2>&1; }



verify_zip_size() {
  local file="$1"
  local min_bytes="$2"
  if [[ ! -f "$file" ]]; then
    echo "ERROR: missing file: $file" >&2
    exit 2
  fi
  local sz
  sz=$(stat -c%s "$file" 2>/dev/null || echo 0)
  if [[ "$sz" -lt "$min_bytes" ]]; then
    echo "ERROR: file looks too small (${sz} bytes): $file" >&2
    echo "This usually means the download was blocked/redirected and you saved an HTML error page." >&2
    echo "Try again with: rm -f '$file' and re-run this script." >&2
    exit 2
  fi
}

verify_zip_integrity() {
  local file="$1"
  if ! unzip -tq "$file" >/dev/null 2>&1; then
    echo "ERROR: zip integrity test failed: $file" >&2
    echo "Delete it and re-download: rm -f '$file'" >&2
    exit 2
  fi
}
download_file() {
  local url="$1"
  local out="$2"

  if [[ -f "$out" ]]; then
    say "Already exists: $out (skip)"
    return 0
  fi

  mkdir -p "$(dirname "$out")"

  if need_cmd wget; then
    say "Downloading (wget): $url -> $out"
    wget -c "$url" -O "$out"
  elif need_cmd curl; then
    say "Downloading (curl): $url -> $out"
    curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"
  else
    echo "ERROR: Need 'wget' or 'curl' installed." >&2
    exit 2
  fi
}

extract_zip() {
  local zip="$1"
  local destdir="$2"

  if [[ ! -f "$zip" ]]; then
    echo "ERROR: zip not found: $zip" >&2
    exit 2
  fi

  if [[ -d "$destdir" ]] && [[ -n "$(ls -A "$destdir" 2>/dev/null || true)" ]]; then
    say "Already extracted: $destdir (skip)"
    return 0
  fi

  if ! need_cmd unzip; then
    echo "ERROR: Need 'unzip' installed." >&2
    exit 2
  fi

  mkdir -p "$(dirname "$destdir")"
  say "Extracting: $zip"
  unzip -q "$zip" -d "$(dirname "$destdir")"
}

main() {
  say "Destination directory: ${DATA_DIR}"
  mkdir -p "${DATA_DIR}"

  mkdir -p "${COCO_DIR}"
  download_file "${VAL_URL}" "${VAL_ZIP}"
  download_file "${ANN_URL}" "${ANN_ZIP}"

  # Quick sanity checks (avoid the common "tiny html file saved as zip" problem)
  verify_zip_size "${VAL_ZIP}" 500000000
  verify_zip_size "${ANN_ZIP}" 100000000
  verify_zip_integrity "${VAL_ZIP}"
  verify_zip_integrity "${ANN_ZIP}"

  if [[ ! -d "${COCO_IMG_DIR}" ]]; then
    extract_zip "${VAL_ZIP}" "${COCO_IMG_DIR}"
  else
    say "COCO images already present: ${COCO_IMG_DIR} (skip)"
  fi

  if [[ ! -f "${COCO_ANN_DIR}/instances_val2017.json" ]]; then
    extract_zip "${ANN_ZIP}" "${COCO_ANN_DIR}"
  else
    say "COCO annotations already present: ${COCO_ANN_DIR}/instances_val2017.json (skip)"
  fi

  download_file "${VTEST_URL}" "${VTEST_PATH}"

  say "Done."
  say "COCO images: ${COCO_IMG_DIR}"
  say "COCO ann:    ${COCO_ANN_DIR}/instances_val2017.json"
  say "Video:       ${VTEST_PATH}"
}

main "$@"
