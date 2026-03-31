from bertopic import BERTopic
from bertopic.representation import VisualRepresentation
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.backend import MultiModalBackend
from PIL import Image, ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import numpy as np
import pandas as pd
from pathlib import Path 


class ImageBertopic:
    def __init__(self, image_ids_path, images_dir):
        self.image_ids = pd.read_csv(image_ids_path)
        self.images_dir = Path(images_dir)
    
    def load_images(self):
        try:
            images = []
            valid_indices = []
            for i, fname in enumerate(self.image_ids["filename"].tolist()):
                try:
                    img_path = self.images_dir / fname
                    if img_path.exists():
                        images.append(str(img_path))
                        valid_indices.append(i)
                except Exception as e:
                    print(f" [skip] {fname}: {e}")
            print(f"Found {len(images)} valid images")
            return images, valid_indices 
        except:
            return [], []
        
    def bertopic_model(self):
        images, valid_indices = self.load_images()
        if len(images) > 0:
            # embeddings = self.embeddings[valid_indices]
            embedding_model = MultiModalBackend('clip-ViT-B-32', batch_size=32)
            representation_model = {
                "Visual_Aspect": VisualRepresentation(image_to_text_model="nlpconnect/vit-gpt2-image-captioning")}
            # representation_model = VisualRepresentation()
            self.topic_model = BERTopic(
                embedding_model=embedding_model,
                representation_model=representation_model,
                min_topic_size=50,
                verbose=True,
                vectorizer_model=CountVectorizer(vocabulary=["image"]),
            )
            topics, probs = self.topic_model.fit_transform(
                documents=None, 
                images = images
            )
            return topics, probs, valid_indices

    
    def main(self):
        topics, probs, valid_indices = self.bertopic_model()
        image_ids_valid = self.image_ids.iloc[valid_indices].copy()
        image_ids_valid["topic"] = topics 
        return topics, probs, image_ids_valid
