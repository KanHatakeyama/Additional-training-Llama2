#!/bin/bash

#GPU id
GPU_ID=0

#電力の初期値
CURRENT_POWER_LIMIT=230

# 電力制限を設定する温度閾値
TEMP_THRESHOLD=69

# 電力制限を設定するワット数の範囲
POWER_LIMIT_MIN=170
POWER_LIMIT_MAX=300
# 電力制限のステップ
POWER_STEP=10

while true; do
    # 現在のGPU温度を取得
    GPU_TEMP=$(nvidia-smi -i $GPU_ID --query-gpu=temperature.gpu --format=csv,noheader,nounits)

    # 温度が閾値を下回るか確認し、電力制限を調整
    if [ "$GPU_TEMP" -lt "$TEMP_THRESHOLD" ]; then
        # 電力制限を増やすが、最大値を超えないようにする
        NEW_POWER_LIMIT=$((CURRENT_POWER_LIMIT + POWER_STEP))
        if [ "$NEW_POWER_LIMIT" -gt "$POWER_LIMIT_MAX" ]; then
            NEW_POWER_LIMIT=$POWER_LIMIT_MAX
        fi
        # 新しい電力制限を適用
        sudo nvidia-smi -i $GPU_ID -pl $NEW_POWER_LIMIT
        CURRENT_POWER_LIMIT=$NEW_POWER_LIMIT

    else
        # 電力制限を減らすが、最小値を下回らないようにする
        NEW_POWER_LIMIT=$((CURRENT_POWER_LIMIT - POWER_STEP))
        if [ "$NEW_POWER_LIMIT" -lt "$POWER_LIMIT_MIN" ]; then
            NEW_POWER_LIMIT=$POWER_LIMIT_MIN
        fi
        # 新しい電力制限を適用
        sudo nvidia-smi -i $GPU_ID -pl $NEW_POWER_LIMIT
        CURRENT_POWER_LIMIT=$NEW_POWER_LIMIT
    fi

    # 一定時間（例えば5秒）ごとにチェック
    sleep 30
done
