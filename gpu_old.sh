#!/bin/bash

# 電力制限を設定する温度閾値
TEMP_THRESHOLD=70
# 電力制限を設定するワット数の範囲
POWER_LIMIT_MIN=190
POWER_LIMIT_MAX=240

while true; do
    for GPU_ID in 0; do
        # 現在のGPU温度を取得
        GPU_TEMP=$(nvidia-smi -i $GPU_ID --query-gpu=temperature.gpu --format=csv,noheader,nounits)

        # 温度が閾値を下回っている場合は、電力制限を最大に設定
        if [ "$GPU_TEMP" -lt "$TEMP_THRESHOLD" ]; then
            sudo nvidia-smi -i $GPU_ID -pl $POWER_LIMIT_MAX
        else
            # 温度が閾値を超えた場合は、電力制限を最小に設定
            sudo nvidia-smi -i $GPU_ID -pl $POWER_LIMIT_MIN
        fi
    done

    # 一定時間（例えば5秒）ごとにチェック
    sleep 60
done
