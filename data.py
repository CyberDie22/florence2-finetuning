import concurrent.futures
import io

import pandas as pd
from datasets import get_dataset_config_names, load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import pyexiv2
import os
from tqdm import tqdm
import csv
import time

allowed_tags = []

with open("danbooru_tags_post_count_general.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[2] == "post_count" or "cosplay" in row[0]:
            continue
        if int(row[2]) > 10:
            allowed_tags.append(row[0])

def generate_grabber_data(path, count_tokens):
    global allowed_tags
    print(f"Loading from path: {path}")
    prompt = []
    answer = []
    image  = []
    for filename in tqdm(os.listdir(path)):
        file_path = os.path.join(path, filename)
        description = get_xmp_description(file_path)
        if description is None:
            continue
        fixed_description = ', '.join([tag for tag in description.split(' ') if tag in allowed_tags])
        
        task_prefix = "<MORE_DETAILED_DANBOORU_PROMPT>"

        #if count_tokens(task_prefix + fixed_description) > 1020:
        #    print(f"File {file_path} has too many tokens!")
        #    time.sleep(5)

        prompt.append(task_prefix)
        answer.append(fixed_description)
        image.append(file_path)
    return prompt, answer, image

def get_xmp_description(path):
    try:
        with pyexiv2.Image(path) as img:
            xmp_data = img.read_xmp()
            if 'Xmp.dc.description' in xmp_data:
                return xmp_data['Xmp.dc.description']['lang="x-default"']
    except Exception as e:
        # ignore
        print(e)
    return None

class GrabberDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example['prompt']
        answer = example['answer']
        image = example['image']
        image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answer, image


class BaseDataset(Dataset):
    def __init__(self, split):
        self._split = split
        self.name = "BaseDataset"
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text


class DocVQADataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.name = "DocVQA"
        self.data = load_dataset("HuggingFaceM4/DocumentVQA", split=split)
        self.task_prompt = "<VQA>"

    def __getitem__(self, idx):
        example = self.data[idx]
        question = self.task_prompt + self.correct_casing_finqa(
            example["question"], True
        )
        first_answer = example["answers"][0]
        answers = self.correct_casing_finqa(first_answer)
        image = example["image"]  # The image is already a PIL Image object
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answers, image
    
class VQAInstructDataset(BaseDataset):
    def __init__(self, split, max_length=1024):
        super().__init__(split)
        self.name = "VQA-Instruct"
        self._max_length = max_length
        self.vqa_data = load_from_disk("/fsx/m4/datasets/complete_single_img_vqa_instruct")
        split_actions = {
                'train': lambda data: data.train_test_split(test_size=0.05, seed=42)['train'],
                'validation': lambda data: data.train_test_split(test_size=0.05, seed=42)['test'].train_test_split(test_size=0.5, seed=42)['train'],
                'test': lambda data: data.train_test_split(test_size=0.05, seed=42)['test'].train_test_split(test_size=0.5, seed=42)['test']
            }

        if split not in split_actions:
            raise ValueError(f"Unknown split: {split}")

        self.vqa_data = split_actions[split](self.vqa_data)
        self.task_prompt = "<VQA>"

    def __len__(self):
        return len(self.vqa_data)
    
    def __getitem__(self, idx):
        example = self.vqa_data[idx]
        texts = random.choice(example['texts'])

        question = self.task_prompt + texts["user"]
        answer = texts["assistant"]

        image = example['images']
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        return question, answer, image

class TheCauldronDataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.name = "The-Cauldron"
        self.images_df, self.texts_df = self.load_all_configs(split)
        self.task_prompt = "<VQA>"

    def __len__(self):
        return len(self.texts_df)
    
    def load_config(self, config_name, split):
        print(f"Loading config: {config_name}")
        dataset = load_dataset("HuggingFaceM4/the_cauldron", config_name, split=split)
        print(f"Finished loading config: {config_name}")

        df_data = dataset.to_pandas()

        # Create the images DataFrame
        df_images = df_data[['images']].copy()
        df_images['image_index'] = df_images.index

        # Explode the texts into separate rows and create a DataFrame
        df_texts = df_data[['texts']].explode('texts').reset_index()
        df_texts.rename(columns={'index': 'image_index'}, inplace=True)

        # Extract 'user', 'assistant', and 'source' from the 'texts' column
        df_texts['question'] = df_texts['texts'].apply(lambda x: x.get('user'))
        df_texts['answer'] = df_texts['texts'].apply(lambda x: x.get('assistant'))
        df_texts['source'] = df_texts['texts'].apply(lambda x: x.get('source'))

        # Drop the original 'texts' column
        df_texts.drop(columns=['texts'], inplace=True)

        # Copy the 'source' column to the images df, using the first source per image index
        df_images = df_images.merge(df_texts[['image_index', 'source']], on='image_index', how='left')
        print(f"Finished processing config: {config_name}")

        return df_images, df_texts

    def load_all_configs(self, split):
        cauldron_config_names = get_dataset_config_names("HuggingFaceM4/the_cauldron")

        images_dfs = []
        texts_dfs = []

        # Use ThreadPoolExecutor for parallel processing and tqdm for progress tracking
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:  # Limit the number of workers
            with tqdm(total=len(cauldron_config_names), desc="Total Progress") as total_pbar:
                futures = {executor.submit(self.load_config, config_name, split): config_name for config_name in cauldron_config_names}
                for future in concurrent.futures.as_completed(futures):
                    config_name = futures[future]
                    try:
                        df_images, df_texts = future.result()
                        images_dfs.append(df_images)
                        texts_dfs.append(df_texts)
                    except Exception as exc:
                        print(f"{config_name} generated an exception: {exc}")
                    total_pbar.update(1)

        # Merge all the loaded DataFrames
        print("Merging DataFrames...")
        merged_images_df = pd.concat(images_dfs, ignore_index=True)
        merged_texts_df = pd.concat(texts_dfs, ignore_index=True)
        print("Finished merging DataFrames")

        return merged_images_df, merged_texts_df

    def __getitem__(self, idx):
        example = self.texts_df.iloc[idx]
        question = example["question"]
        answer = example["answer"]
        source = example["source"]
        image_idx = example["image_index"]

        image_data = self.images_df.loc[(self.images_df['image_index'] == image_idx) & (self.images_df['source'] == source), 'images'].values[0][0]['bytes'] 
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        return question, answer, image
