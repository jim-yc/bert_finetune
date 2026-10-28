"""
prepare_data.py

把原始 data.csv 轉成 BERT 八元分類訓練所需的 train / validation JSONL。

這支程式做的事情：

    data.csv
        ↓
    讀取文字與 label
        ↓
    檢查格式
        ↓
    以 label 做 stratified train / validation split
        ↓
    data/bert/train.jsonl
    data/bert/valid.jsonl

為什麼要先獨立做資料切分？

因為之後 train_bert.py 每次執行時，都直接讀取固定的 train / validation 檔案，
比較不同模型或超參數時，不會因為每次重新亂數切分而改變資料集。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

import csv
import json
from collections import Counter # 用來統計 label 0 / 1 各有多少筆。
from pathlib import Path

from sklearn.model_selection import train_test_split

# =============================================================================
# 2. 資料設定
# =============================================================================

INPUT_FILE = Path("data.csv")
OUTPUT_DIR = Path("data/bert")
VALID_RATIO = 0.2
SEED = 42
LABEL2ID = {
    "平淡語氣": 0,
    "開心語調": 1,
    "關切語調": 2,
    "憤怒語調": 3,
    "驚奇語調": 4,
    "悲傷語調": 5,
    "厭惡語調": 6,
    "疑問語調": 7,
}

# =============================================================================
# 3. 輔助函式
# =============================================================================

def load_data_csv(path: Path):
    """
    讀取 data.csv。

    回傳：
        texts  : list[str]
        labels : list[int]

    我們預期 label 只能是整數0-7
    """

    texts = []
    labels = []

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row_dict in csv_reader:
            texts.append(row_dict["text"])
            labels.append(LABEL2ID[row_dict["emotion"]])

    if not texts:
        raise RuntimeError(f"{path} 沒有讀到任何有效資料。")

    return texts, labels


def save_jsonl(texts, labels, path: Path):
    """
    將文字與標籤寫成 JSONL。

    每一行格式：
        {"sentences": "這間飯店很好", "labels": 1}
    """

    with path.open("w", encoding="utf-8") as file:
        for text, label in zip(texts, labels):
            row = {"sentences": text, "labels": label}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_label_distribution(name, labels):
    """
    顯示資料量與比例。
    """

    counts = Counter(labels)
    total = len(labels)
    print(f"{name}：{total} 筆")

    for label in sorted(counts):
        count = counts[label]
        ratio = count / total
        print(f"    label= {label}: {count} 筆 ({ratio:.2%})")

# =============================================================================
# 4. 主流程
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Step 1：確認原始資料存在
    # -------------------------------------------------------------------------

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"找不到原始資料：{INPUT_FILE.resolve()}")

    # -------------------------------------------------------------------------
    # Step 2：讀取 data.csv
    # -------------------------------------------------------------------------

    texts, labels = load_data_csv(INPUT_FILE)

    print("=" * 72)
    print_label_distribution("全部資料", labels)

    # -------------------------------------------------------------------------
    # Step 3：切 train / validation
    # -------------------------------------------------------------------------

    # stratify 會讓兩邊盡量維持相似的 class distribution。
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts,
        labels,
        test_size=VALID_RATIO,
        random_state=SEED,
        stratify=labels,
    )

    # -------------------------------------------------------------------------
    # Step 4：建立輸出資料夾
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    # -------------------------------------------------------------------------
    # Step 5：儲存 JSONL
    # -------------------------------------------------------------------------

    save_jsonl(train_texts, train_labels, train_path)
    save_jsonl(valid_texts, valid_labels, valid_path)

    # -------------------------------------------------------------------------
    # Step 6：顯示結果
    # -------------------------------------------------------------------------

    print_label_distribution("Training", train_labels)
    print_label_distribution("Validation", valid_labels)

    print(f"Training JSONL：{train_path}")
    print(f"Validation JSONL：{valid_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
