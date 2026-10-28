"""
infer_bert.py

載入 train_bert.py 儲存的最佳 BERT classifier，
對新的繁體中文評論進行情感分類。
"""

from pathlib import Path
from pprint import pprint

import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# =============================================================================
# 1. 推論設定
# =============================================================================

MODEL_DIR = Path("output/bert_binary_clf")

# 可以直接修改這個 list 測試不同句子。
TEST_TEXTS = [
    "這個房間真的不錯，服務人員也很親切，下次還會再來！",
    "這個房間真的很爛，服務人員也很差，下次不會再來！",
    "一般般",
]

def get_gpu_type():
    """
    顯示目前 PyTorch 是否看到 GPU。
    """

    print("=" * 72)
    print(f"PyTorch：{torch.__version__}")
    print(f"Transformers：{transformers.__version__}")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_type = torch.device("xpu")
        print(f"偵測到 Intel GPU (XPU): {torch.xpu.get_device_name(0)}")

    elif torch.cuda.is_available():
        device_type = torch.device("cuda")
        print(f"偵測到 NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0)}")

    else:
        device_type = torch.device("cpu")
        print(f"沒偵測到任何 GPU")

    print("=" * 72)
    return device_type

# =============================================================================
# 2. 主流程
# =============================================================================

def main():

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到 fine-tuned model：{MODEL_DIR.resolve()}\n"
            "請先執行 train_bert.py。"
        )

    # -------------------------------------------------------------------------
    # Step 1：載入 tokenizer
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # -------------------------------------------------------------------------
    # Step 2：載入 sequence classifier
    # -------------------------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # -------------------------------------------------------------------------
    # Step 3：決定 inference device
    # -------------------------------------------------------------------------

    device = get_gpu_type()
    
    # -------------------------------------------------------------------------
    # Step 4：建立 Hugging Face pipeline
    # -------------------------------------------------------------------------

    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Step 5：執行分類
    # -------------------------------------------------------------------------

    results = classifier(
        TEST_TEXTS,
        truncation=True,
        max_length=512,
    )

    # -------------------------------------------------------------------------
    # Step 6：顯示結果
    # -------------------------------------------------------------------------

    for text, result in zip(TEST_TEXTS, results):
        print("=" * 72)
        print(f"文字：{text}")
        print(f"預測：{result['label']}")
        print(f"信心分數：{result['score']:.6f}")

    print("=" * 72)

    # 如果想直接查看原始 list，也可以保留 pprint。
    print("Raw pipeline output：")
    pprint(results)


if __name__ == "__main__":
    main()
