import os
import cv2
import lmdb
import numpy as np
import pickle

def create_dummy_lmdb(output_dir, split_name, num_samples=5):
    lmdb_path = os.path.join(output_dir, 'lmdb', 'refcoco', split_name + '.lmdb')
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    
    # Also create masks directory as the dataset loader looks for actual .png files
    mask_dir = os.path.join(output_dir, 'masks', 'refcoco')
    os.makedirs(mask_dir, exist_ok=True)
    
    print(f"Creating dummy LMDB at {lmdb_path} with {num_samples} samples...")
    env = lmdb.open(lmdb_path, map_size=1073741824) # 1GB
    
    keys = []
    with env.begin(write=True) as txn:
        for i in range(num_samples):
            seg_id = i + 1
            
            # 1. Create a dummy image (RGB)
            img = np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8)
            _, img_encoded = cv2.imencode('.jpg', img)
            
            # 2. Create a dummy ground truth mask and save it as .png
            mask = np.zeros((320, 320), dtype=np.uint8)
            cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)  # simple white square
            mask_path = os.path.join(mask_dir, f"{seg_id}.png")
            cv2.imwrite(mask_path, mask)
            
            # 3. Create dummy sentences
            sents = ["the white square", "a box on the left"]
            
            # 4. Pack into the dict structure CRIS expects 
            # Note: the dataset loader also needs mask in lmdb for training mode
            _, mask_encoded = cv2.imencode('.png', mask)
            
            ref = {
                'img': img_encoded.tobytes(),
                'seg_id': seg_id,
                'sents': sents,
                'num_sents': len(sents),
                'mask': mask_encoded.tobytes()
            }
            
            # Serialize
            buf = pickle.dumps(ref)
            
            key = f"{seg_id}".encode('ascii')
            keys.append(key)
            txn.put(key, buf)
            
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))
        
    env.close()
    print(f"Generated {split_name} successfully.")

if __name__ == '__main__':
    dataset_root = "datasets"
    create_dummy_lmdb(dataset_root, 'train', num_samples=10)
    create_dummy_lmdb(dataset_root, 'val', num_samples=5)
    print("Dummy dataset creation complete!")
