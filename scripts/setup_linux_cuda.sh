#!/usr/bin/env bash
set -euo pipefail
mkdir -p vendor models
if [ ! -d vendor/qwentts.cpp ]; then
  git clone https://github.com/ServeurpersoCom/qwentts.cpp vendor/qwentts.cpp
fi
cmake -S vendor/qwentts.cpp -B vendor/qwentts.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/qwentts.cpp/build --config Release -j --target qwen-tts
