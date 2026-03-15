"""
生成训练数据 - 使用COCO图像创建RefCOCO格式的LMDB数据
"""
import os
import cv2
import lmdb
import pickle
import numpy as np
from tqdm import tqdm

# COCO图像目录
COCO_DIR = "datasets/images/train2014"
OUTPUT_DIR = "datasets/lmdb/refcoco_train"

# 示例文本描述
SAMPLE_SENTS = [
    "the cat on the left",
    "the person wearing blue",
    "a red chair",
    "the tall building",
    "the white dog",
    "a man in the background",
    "the table near the window",
    "the child playing",
    "the woman with glasses",
    "the black car"
]

def create_sample_data(num_samples=1000):
    """创建训练数据"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取COCO图像列表
    img_files = [f for f in os.listdir(COCO_DIR) if f.endswith('.jpg') or f.endswith('.png')]
    print(f"找到 {len(img_files)} 张COCO图像")

    # 创建LMDB
    env = lmdb.open(OUTPUT_DIR, map_size=10*1024*1024*1024)  # 10GB

    data_list = []
    keys = []

    with env.begin(write=True) as txn:
        for i, img_file in enumerate(tqdm(img_files[:num_samples], desc="生成数据")):
            img_path = os.path.join(COCO_DIR, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 随机选择区域生成mask
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            # 随机生成1-3个区域
            num_objs = np.random.randint(1, 4)
            for _ in range(num_objs):
                x1 = np.random.randint(0, max(1, w-50))
                y1 = np.random.randint(0, max(1, h-50))
                x2 = np.random.randint(x1+20, min(w, x1+200))
                y2 = np.random.randint(y1+20, min(h, y1+200))
                mask[y1:y2, x1:x2] = 255

            # 随机选择文本描述
            num_sents = np.random.randint(1, 4)
            sents = np.random.choice(SAMPLE_SENTS, num_sents, replace=False).tolist()

            # 编码图像
            _, img_encoded = cv2.imencode('.jpg', img)
            img_bytes = img_encoded.tobytes()

            # 编码mask
            _, mask_encoded = cv2.imencode('.png', mask)
            mask_bytes = mask_encoded.tobytes()

            data = {
                'img': img_bytes,
                'mask': mask_bytes,
                'seg_id': i + 1,
                'sents': sents,
                'num_sents': len(sents)
            }

            key = str(i + 1).encode()
            keys.append(key)
            txn.put(key, pickle.dumps(data))

        # 保存元数据
        txn.put(b'__len__', pickle.dumps(len(keys)))
        txn.put(b'__keys__', pickle.dumps(keys))

    env.close()
    print(f"生成完成: {len(keys)} 个样本")
    print(f"保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    create_sample_data(5000)  # 生成5000个训练样本
