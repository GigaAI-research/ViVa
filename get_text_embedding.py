import os
import pdb
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from wan.modules.t5 import T5EncoderModel


class TxtDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: list of tuples (txt_path, save_pth)
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        txt_path, save_pth = self.data_list[idx]
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()  
            if not content:
                content = "A two-hand robot is performing a task with grippers."
            else:
                content = " ".join(content.split())
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            content = ""

        return content, save_pth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txt_list",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
    )


    args = parser.parse_args()
    data_list = []
    save_base = 'data/t5_embedding'
    video_num = 0
    with open(args.txt_list, 'r') as file:
        for line in file:
            if len(line.strip()) != 0:
                txt_path = line.strip()
                txt_name = txt_path.split('/')[-1]
                sub_dataset = txt_path.split('/')[-2]
                data_set = txt_path.split('/')[-3]
                save_dir = os.path.join(save_base, data_set, sub_dataset)
                os.makedirs(save_dir, exist_ok=True)
                save_pth = os.path.join(save_dir, txt_name.split('.')[0] + '.pt')
                data_list.append((txt_path, save_pth))
                video_num += 1
                print(video_num)
    print(f"Total files to process: {len(data_list)}")

    device = args.device
    wan_path = 'weights/Wan2.2-TI2V-5B'
    ckpt = os.path.join(wan_path, "models_t5_umt5-xxl-enc-bf16.pth")
    tok = os.path.join(wan_path, "google/umt5-xxl")
    print(f"Loading T5 model to {device}...")
  
    t5_model = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=ckpt,
        tokenizer_path=tok,
    )

    t5_model.model.eval()
    dataset = TxtDataset(data_list)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,  
        num_workers=12,
        pin_memory=True,  
        drop_last=False  
    )

    print("Starting inference...")
    with torch.no_grad():  
        for batch_texts, batch_save_paths in tqdm(dataloader):
            texts_list = list(batch_texts)
            try:

                # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                context = t5_model(texts_list, device)
                if isinstance(context, torch.Tensor):
                    context = context.cpu()
                elif isinstance(context, list):  
                    context = [c.cpu() for c in context]


                for i, emb in enumerate(context):
                    torch.save(emb.clone(), batch_save_paths[i])

            except Exception as e:
                print(f"Error processing batch starting with {batch_save_paths[0]}: {e}")
                continue


if __name__ == "__main__":
    main()



