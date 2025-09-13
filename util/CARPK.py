import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class CARPK(Dataset):
    def __init__(self, img_size=(384, 683), root_path='/media/cs4007/adfc2692-0951-4a9b-8ea6-ebfa0e11323b/wwj/datasets/CARPK_devkit/data/'):
        self.img_size = img_size
        super().__init__()
        self.imglist = []
        with open(os.path.join(root_path, 'ImageSets/test.txt'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                dic = line.strip()
                self.imglist.append(dic)
        with open("/media/cs4007/adfc2692-0951-4a9b-8ea6-ebfa0e11323b/wwj/CLIPCAC/text_embedding/clip_text_embedding_oc.json", "r") as f:
            self.text_tokens = json.load(f)

        self.root_path = root_path
        self.resize = T.Resize(img_size)

    def __getitem__(self, index):
        imginfo = self.imglist[index]

        img = Image.open(os.path.join(self.root_path, 'Images/', imginfo + '.png')).convert("RGB")
        re_w, re_h = self.img_size[0], self.img_size[1]
        w, h = img.size
        img = T.Compose([
            T.ToTensor(),
            self.resize,
        ])(img)

        anno = open(
            os.path.join(self.root_path, 'Annotations', imginfo + '.txt'), 'r', encoding='utf-8'
        )
        all_bboxes = []
        for line in anno.readlines():
            box = line.split()[:-1]
            box = [int(b) for b in box]
            all_bboxes.append(box)
        dots = len(all_bboxes)
        sample = {'image': img, 'dots': dots}
        sample['text'] = torch.tensor(self.text_tokens['6900.jpg'])
        return sample['image'], sample['text'], sample['dots'], imginfo

    def __len__(self):
        return len(self.imglist)
