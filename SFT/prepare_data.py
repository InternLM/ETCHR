import json
import io
from PIL import Image
from datasets import Dataset, Sequence, Features, Value
from datasets import Image as ImageData
import os
import pandas as pd

ds = pd.read_parquet('your_parquet')
total_len = len(ds)
for i in range(total_len):
    image_bytes = ds.iloc[i]["source_image"][0]['bytes']
    image = Image.open(io.BytesIO(image_bytes))
    save_path = ds.iloc[i]["source_image"][0]['path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

