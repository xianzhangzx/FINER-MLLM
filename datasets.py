import os
from PIL import Image
import random
import json
import re
from torch.utils.data import Dataset
from lavis.processors.blip_processors import BlipCaptionProcessor

class CLEVR_Dataset(Dataset):
    def __init__(self, data_path, transform, split='train', max_words=40, prompt="", scale=1.0, rag_store_path=None, k=5):
        super().__init__()
        self.data_path = data_path
        self.default_image_path = os.path.join(self.data_path, 'images')
        self.nsc_image_path = os.path.join(self.data_path, 'nsc_images')
        self.sc_image_path = os.path.join(self.data_path, 'sc_images')
        self.split = split
        self.transform = transform
        self.max_words = max_words
        self.prompt = prompt
        self.scale = scale
        self.text_process = BlipCaptionProcessor(prompt=prompt, max_words=max_words)
        self.word_list = None
        self.rag_store = json.load(open(rag_store_path,'rb'))
        self.k = k
        
        with open(os.path.join(self.data_path, "splits.json"), 'r') as fp:
            total_image_ids = json.load(fp)
            self.image_ids = total_image_ids[split]

        with open(os.path.join(self.data_path, "change_captions.json"), 'r') as fp:
            self.change_captions = json.load(fp)

        with open(os.path.join(self.data_path, "no_change_captions.json"), 'r') as fp:
            self.no_change_captions = json.load(fp)

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_name = "CLEVR_default_%06d.png" % int(image_id)
        sc_caption = random.choice(self.change_captions[image_name])
        nsc_caption = random.choice(self.no_change_captions[image_name])

        bef_image_path = os.path.join(self.default_image_path, image_name)
        aft_image_path = os.path.join(self.sc_image_path, image_name.replace('default', 'semantic'))
        no_image_path = os.path.join(self.nsc_image_path, image_name.replace('default', 'nonsemantic'))

        bef_image_data = self.get_image_data(bef_image_path)
        aft_image_data = self.get_image_data(aft_image_path)
        no_image_data = self.get_image_data(no_image_path)

        sc_reference_words = self.convert_captions_to_lexicon("sc_"+image_name)
        nsc_reference_words = self.convert_captions_to_lexicon("nsc_"+image_name)
        
        out = {}
        out['sc_ref_words'] = sc_reference_words
        out['nsc_ref_words'] = nsc_reference_words

        if self.split == 'train':
            out['sc_caption'] = self.text_process(sc_caption)
            out['nsc_caption'] = self.text_process(nsc_caption)
            if random.random() < 0.5:
                out['bef_image'] = bef_image_data
                out['aft_image'] = aft_image_data
                out['nsc_image'] = no_image_data
            else:
                out['bef_image'] = no_image_data
                out['aft_image'] = aft_image_data
                out['nsc_image'] = bef_image_data
        else:
            out['bef_image'] = bef_image_data
            out['aft_image'] = aft_image_data
            out['nsc_image'] = no_image_data
            out['img_id'] = "%06d.png" % int(image_id)
            out['bef_path'] = bef_image_path
            out['aft_path'] = aft_image_path
            out['nsc_path'] = no_image_path
        return out
    
    def get_image_data(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        # raw image size:(320,480,3)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def convert_captions_to_lexicon(self, image_id):
        k = self.k
        relevant_setences = self.rag_store[image_id]
        sentences = relevant_setences[:k]
        all_words = " ".join(sentences).split(' ')
        filter_words = []
        for w in all_words:
            if w not in filter_words:
                filter_words.append(w)
        return " ".join(filter_words)
    
class SpotDataset(Dataset):
    def __init__(self, image_path, anno_path, racap_path, transform=None, split='train', prompt="", k=5) -> None:
        super().__init__()
        self.image_path = image_path
        self.split = split
        self.transform = transform
        self.text_preprocess = BlipCaptionProcessor(prompt=prompt, max_words=40)
        captions_file_path = os.path.join(anno_path,'filter_{}.json'.format(split))
        with open(captions_file_path,'r') as f:
            self.captions = json.load(f)
        with open(racap_path, 'r') as f:
            self.rag_captions = json.load(f)
        self.k = k

        with open(os.path.join(anno_path, 'filter_train.json'), 'r') as f:
            training_captions = json.load(f)
        self.texts = []
        for cap in training_captions:
            for sentence in cap['sentences']:
                self.texts.append(self.text_preprocess(sentence))
           
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = caption['img_id']
        text = random.choice(caption['sentences'])
        text = self.text_preprocess(text)
        bef_img = self.get_image(img_id)
        aft_img = self.get_image(img_id+'_2')
        
        assert img_id in self.rag_captions.keys(), "image %s does not has retrieval captions" % img_id
        ref_words = self.convert_captions_to_lexicon(self.rag_captions[img_id])
        
        output = {}
        if self.split == 'train':
            output['bef_img'] = bef_img
            output['aft_img'] = aft_img
            output['caption'] = text
            output['lexicon'] = ref_words
        else:
            output['bef_path'] = os.path.join(self.image_path, '%s.png' % img_id)
            output['aft_path'] = os.path.join(self.image_path, '%s.png' % (img_id+"_2"))
            output['bef_img'] = bef_img
            output['aft_img'] = aft_img
            output['img_id'] = "%s.png" % img_id
            output['lexicon'] = ref_words
            
        return output
    
    def get_image(self, img_id):
        img_path = os.path.join(self.image_path, '%s.png' % img_id)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def convert_captions_to_lexicon(self, racaps):
        sentences = racaps[:self.k]
        all_words = " ".join(sentences).split(' ')
        filter_words = []
        for w in all_words:
            if w not in filter_words:
                filter_words.append(w)
        return " ".join(filter_words)
    
class Image_Edit_Request(Dataset):  
    def __init__(self, data_path, transform, split="split", max_words=40, prompt="", rag_store_path=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.image_path = os.path.join(data_path, "images")
        self.split = split
        self.transform = transform
        self.max_words = max_words
        self.prompt = prompt
        self.text_process = BlipCaptionProcessor(prompt=prompt, max_words=max_words)
        self.word_list = None
        self.rag_captions = None
        if rag_store_path:
            with open(rag_store_path, "r") as f:
                self.rag_captions = json.load(f)

        with open(os.path.join(self.data_path, "%s.json"%self.split), "r") as f:
            self.captions = json.load(f)

    def __len__(self,):
        return len(self.captions)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        text = random.choice(caption['sents'])
        text = self.text_process(text)
        bef_img = self.get_image(caption['img0'])
        aft_img = self.get_image(caption['img1'])
        img_id = caption['uid']
        if self.rag_captions is not None:
            ref_words = self.convert_captions_to_lexicon(self.rag_captions[img_id], k=5)
        else:
            ref_words = None
        out = {}
        if self.split == 'train':
            out['bef_img'] = bef_img
            out['aft_img'] = aft_img
            out['caption'] = text
            out['lexicon'] = ref_words if ref_words else "1"
        else:
            out['bef_img'] = bef_img
            out['aft_img'] = aft_img
            out['img_id'] = img_id
            out['lexicon'] = ref_words if ref_words else "1"
            out['bef_path'] = os.path.join(self.image_path, caption['img0'])
            out['aft_path'] = os.path.join(self.image_path, caption['img1'])
        return out
    
    def get_image(self, image_name):
        img_path = os.path.join(self.image_path, image_name)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def convert_captions_to_lexicon(self, racaps, k=3):
        all_words = " ".join(racaps[:k]).split(' ')
        filter_words = []
        for w in all_words:
            if w not in filter_words:
                filter_words.append(w)
        return " ".join(filter_words)
    

