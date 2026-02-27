#!/bin/bash
#SBATCH --job-name=nemotron-nano-test   # <= 請改成你的 job name (識別用)
#SBATCH --partition=dev                 #    測試用請使用 dev
#SBATCH --nodes=1                       #    請改成你的節點數量，normal partition 最少 2 個節點
#SBATCH --ntasks-per-node=1             #    每台節點的 task 數量
#SBATCH --gres=gpu:H200:8               #    每台節點的 GPU 數量
#SBATCH --cpus-per-task=104             #    每台節點的 CPU 數量
#SBATCH --mem=0                         #    設置為該節點所有記憶體
#SBATCH --time=01:30:00                 #    測試用不會太長
#SBATCH --output=logs/slurm_test_%j.out #    輸出日誌
#SBATCH --error=logs/slurm_test_%j.err  #    錯誤日誌
#SBATCH --account=YOUR_ACCOUNT          # <= 請改成你的帳號或計畫代號

set -euo pipefail

# ==============================================================================
# 使用者定義輸入 (User defined Inputs) - 可透過執行腳本時傳入參數覆蓋
# ------------------------------------------------------------------------------
# 1. MODEL_NAME: 欲進行評測的模型名稱。可以是 Hugging Face 上的 Repo ID 
#                (例如 "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16") 或本地端絕對路徑。
MODEL_NAME=${1:-"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"}

# 2. TP_SIZE (Tensor Parallel Size): 決定模型要在單一節點內切分給「幾張 GPU」同時運算。
#                                    若模型超過單卡 VRAM，請調高此數值。
TP_SIZE=${2:-1}

# 3. PP_SIZE (Pipeline Parallel Size): 管線平行處理大小。一般針對超大模型才需設定，預設為 1 即可。
PP_SIZE=${3:-1}

# 4. HF_UPLOAD_REPO: Hugging Face 上用來存放評測結果資料集的 Repo 路徑。
#                    請確保系統環境已登入 Hugging Face (執行過 `hf auth login`) 且具備寫入權限。
HF_UPLOAD_REPO=${4:-"whats2000/nemotron-nano-test-logs-and-scores"}

# 5. 生成控制參數 (Generation Parameters):
MAX_TOKENS=${5:-4096}
THINKING_START_TAG=${6:-"<think>"}
THINKING_END_TAG=${7:-"</think>"}

# 6. 伺服器啟動參數 (Server Parameters):
#    MAX_MODEL_LEN: 模型最大上下文長度 (會影響 vLLM 分配的 GPU 記憶體量)
MAX_MODEL_LEN=${8:-32768}

# 9. HF_VARIANT: 上傳到 Hugging Face 時的 variant 名稱，用於在同一個 Repo 下區分不同評測條件。
#               例如："test_calibration"、"high-temperature"、"baseline" 等。
HF_VARIANT=${9:-"test_calibration"}

# 7. 工作目錄 (Working Directory): 請設定為您存放此專案程式碼的絕對路徑。
WORK_DIR="/home/whats2000/workspace/Eval"      # <= 請改成你的工作目錄

# ------------------------------------------------------------------------------
# ⚠️ 注意 (Notice):
# 其他如資料集設定 (dataset_paths)、System Prompt 或是較少更動的評測選項，
# 仍需要直接去修改 yaml 配置檔 (configs/config_slurm_test.yaml)。
# ==============================================================================

mkdir -p logs results

# 確保運算節點上有使用者本地二進制檔 (uv, hf) 於路徑中
export PATH="$HOME/.local/bin:$HOME/miniconda3/bin:$HOME/miniconda3/condabin:$PATH"

cd "${WORK_DIR}"

# 將 HF/模型快取導向 /work 以避免 /home 空間配額耗盡
export HF_HOME="/work/${USER}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export UV_CACHE_DIR="/work/${USER}/.cache/uv"
mkdir -p "${HF_HOME}"

# Token: 透過 sbatch --export 傳入；若無則自動讀取 token 快取檔案
if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN="$(cat "${HF_HOME}/token" 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null || true)"
fi
: "${HF_TOKEN:?ERROR: HF_TOKEN not set. Run: hf auth login}"

export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

VLLM_VENV="${WORK_DIR}/vllm-venv"
if [[ ! -x "${VLLM_VENV}/bin/python" ]]; then
    echo "ERROR: vllm venv not found at ${VLLM_VENV}" >&2
    exit 1
fi

# 計算動態變數
NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
INSTANCES_PER_NODE=$(( GPUS_PER_NODE / (TP_SIZE * PP_SIZE) ))

# 環境資訊
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | xargs)
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
GPU_MEMORY_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0 | awk '{print $1}')
GPU_MEMORY=$(echo "scale=1; $GPU_MEMORY_MIB / 1024" | bc 2>/dev/null || awk "BEGIN {printf \"%.1f\", $GPU_MEMORY_MIB/1024}")
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 | xargs)
PYTHON_VERSION=$(uv run python --version | awk '{print $2}')
TORCH_VERSION=$("${VLLM_VENV}/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")

# ==============================================================================
# 準備資料集 (Dataset Preparation)
# ------------------------------------------------------------------------------
DATASET_DIR="/work/${USER}/datasets"
mkdir -p "${DATASET_DIR}"

if [[ ! -d "${DATASET_DIR}/lianghsun__tw-legal-benchmark-v1" ]]; then
    echo "正在下載 tw-legal-benchmark-v1 ..."
    uv run twinkle-eval --download-dataset lianghsun/tw-legal-benchmark-v1 \
        --dataset-split train \
        --output-dir "${DATASET_DIR}"
fi
# ==============================================================================

# 建立動態配置檔
CONFIG_RUN="configs/run_test_${SLURM_JOB_ID:-test}.yaml"
cp configs/config_slurm_test.yaml.yaml $CONFIG_RUN 2>/dev/null || true
cp configs/config_slurm_test.yaml $CONFIG_RUN

sed -i "s|PLACEHOLDER_MODEL_NAME|${MODEL_NAME}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_DATASET_DIR|${DATASET_DIR}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_GPU_MODEL|${GPU_MODEL}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_GPU_COUNT|${GPU_COUNT}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_GPU_MEMORY_GB|${GPU_MEMORY}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_CUDA_VERSION|${CUDA_VERSION}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_DRIVER_VERSION|${DRIVER_VERSION}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_TP_SIZE|${TP_SIZE}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_PP_SIZE|${PP_SIZE}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_PYTHON_VERSION|${PYTHON_VERSION}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_TORCH_VERSION|${TORCH_VERSION}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_NNODES|${NNODES}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_MAX_MODEL_LEN|${MAX_MODEL_LEN}|g" $CONFIG_RUN

# 如果 tags 是字串 "null"，在 YAML 中應該要是 null 關鍵字，我們可以根據變數值決定替換格式
if [ "$THINKING_START_TAG" = "null" ]; then
    sed -i "s|PLACEHOLDER_THINKING_START_TAG|null|g" $CONFIG_RUN
else
    sed -i "s|PLACEHOLDER_THINKING_START_TAG|\"${THINKING_START_TAG}\"|g" $CONFIG_RUN
fi

if [ "$THINKING_END_TAG" = "null" ]; then
    sed -i "s|PLACEHOLDER_THINKING_END_TAG|null|g" $CONFIG_RUN
else
    sed -i "s|PLACEHOLDER_THINKING_END_TAG|\"${THINKING_END_TAG}\"|g" $CONFIG_RUN
fi

sed -i "s|PLACEHOLDER_MAX_TOKENS|${MAX_TOKENS}|g" $CONFIG_RUN

echo "============================================="
echo " 開始校準測試 "
echo " 模型 (MODEL): $MODEL_NAME"
echo " 策略 (STRATEGY): TP=$TP_SIZE | PP=$PP_SIZE | 實例數/節點=$INSTANCES_PER_NODE"
echo "============================================="

START_TIME=$(date +%s)

for i in $(seq 0 $((INSTANCES_PER_NODE - 1))); do
    START_GPU=$(( i * TP_SIZE * PP_SIZE ))
    END_GPU=$(( START_GPU + TP_SIZE * PP_SIZE - 1 ))
    GPU_LIST=$(seq -s ',' $START_GPU $END_GPU)
    PORT=$(( 8000 + i ))
    BASE_URL="http://localhost:${PORT}/v1"
    
    LOCAL_CONFIG="${CONFIG_RUN%.yaml}_rank${i}.yaml"
    cp "${CONFIG_RUN}" "$LOCAL_CONFIG"
    sed -i "s|PLACEHOLDER_BASE_URL|${BASE_URL}|g" "$LOCAL_CONFIG"
    
    (
        export CUDA_VISIBLE_DEVICES=$GPU_LIST
        export RANK=$i
        export WORLD_SIZE=$INSTANCES_PER_NODE

        echo "[Rank $i] 啟動 vLLM (GPUs: $GPU_LIST, Port: $PORT)..."
        "${VLLM_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
            --model "${MODEL_NAME}" \
            --tensor-parallel-size "${TP_SIZE}" \
            --pipeline-parallel-size "${PP_SIZE}" \
            --port "${PORT}" \
            --gpu-memory-utilization 0.90 \
            --max-model-len "${MAX_MODEL_LEN}" \
            --trust-remote-code > "logs/vllm_test_rank${i}.log" 2>&1 &
        VLLM_PID=$!
        
        timeout 3600 bash -c "until curl -s ${BASE_URL}/models > /dev/null; do sleep 5; done" || {
            echo "[Rank $i] vLLM 伺服器啟動失敗。"
            kill $VLLM_PID
            exit 1
        }
        
        echo "[Rank $i] vLLM 伺服器已就緒。開始評測 (本機寫入)..."
        uv run twinkle-eval --config "$LOCAL_CONFIG" --export json
        
        kill $VLLM_PID
        rm -f "$LOCAL_CONFIG"
    ) &
done

wait
rm -f $CONFIG_RUN

# ==========================================
# 5. 合併評測結果與上傳
# ==========================================
echo "等待檔案系統同步..."
sleep 5

# 同時搜尋分散式碎片與單節點結果，取最新時間戳記
LATEST_TIMESTAMP=$(
    { ls -1qr results/results_????????_????.json 2>/dev/null || true; } \
    | head -n 1 | grep -oP '\d{8}_\d{4}' || true
)

if [ -n "$LATEST_TIMESTAMP" ]; then
    echo "=========================================="
    echo "取得評測結果 (Timestamp: $LATEST_TIMESTAMP)，開始合併與上傳..."

    if [ -n "$HF_UPLOAD_REPO" ] && [ "$HF_UPLOAD_REPO" != "your-org/eval-logs" ]; then
        uv run twinkle-eval --finalize-results "${LATEST_TIMESTAMP}" --hf-repo-id "${HF_UPLOAD_REPO}" --hf-variant "${HF_VARIANT}"
    else
        uv run twinkle-eval --finalize-results "${LATEST_TIMESTAMP}"
    fi
else
    echo "⚠️ 找不到任何可供處理的評測結果，請確認評測是否已完成並寫入 results/ 目錄。"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "評測完成，共耗時 $DURATION 秒。"

echo "校準總結:"
echo "耗時: $DURATION 秒"
echo "請使用此結果來調整 TP_SIZE 和 DP 節點數量，以確保總運行時間在您的限制範圍內。"
