import clip 
import torch 
import numpy as np 
import pandas as pd 
from PIL import Image, ImageFile
from pathlib import Path 
from tqdm import tqdm 

ImageFile.LOAD_TRUNCATED_IMAGES = True 
device     = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model.eval()

BATCH_SIZE = 64

class ImageEmbedding:
    def __init__(self):
        pass

    def get_image_embeddings(self, image_files, output_path, file_name, csv_name):
        all_embeddings = []
        all_filenames  = []

        for batch_start in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Embedding"):
            batch_files = image_files[batch_start: batch_start + BATCH_SIZE]
            tensors     = []

            for f in batch_files:
                try:
                    img = Image.open(f).convert("RGB")
                    tensors.append(preprocess(img))
                    all_filenames.append(f.name)
                except Exception as e:
                    print(f"  [skip] {f.name}: {e}")
                    continue

            if not tensors:
                print("Not tensors")
                continue

            batch_tensor = torch.stack(tensors).to(device)
            with torch.no_grad():
                embs = model.encode_image(batch_tensor)
                embs = embs / embs.norm(dim=-1, keepdim=True)

            all_embeddings.append(embs.cpu().numpy())
        
        print("Length of embeddings list: ", len(all_embeddings))
        embeddings_matrix = np.vstack(all_embeddings)
        print("Embeddings matrix:", type(embeddings_matrix))

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / file_name, embeddings_matrix)
        pd.DataFrame({"filename": all_filenames}).to_csv(
            out_dir / csv_name, index=False
        )
        return embeddings_matrix, out_dir
        
    def main(self, image_ids, folder_path, output_path, file_name, csv_name):
        image_files = sorted([
            f for f in Path(folder_path).iterdir()
            if f.name in image_ids])
        print(f"Found {len(image_files)} images")
        embeddings_matrix, out_dir = self.get_image_embeddings(image_files, output_path, file_name, csv_name)
        print(f"Saved {embeddings_matrix.shape} embeddings → {out_dir}")
    

