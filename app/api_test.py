import os 
import random
import time

from glob import glob 
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import numpy as np

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification, DetrImageProcessor, DetrForObjectDetection
import torch
from datasets import load_dataset
import timm
import requests
from torchvision import transforms


import numpy as np
import gradio as gr

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")



def predict(img):
    im_copy = img.copy()
    img = Image.fromarray(img)
    cls_inputs = feature_extractor(im_copy, return_tensors="pt")
    cls_inputs = cls_inputs.to(device)
    print(type(img))
    with torch.no_grad():
        logits = cls_model(**cls_inputs).logits
        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        if cls_model.config.id2label[predicted_label] == 'tray':
            cls_model.config.id2label[predicted_label] = 'apple'
        cls_text = cls_model.config.id2label[predicted_label]
    
    # obj runs
    obj_inputs = processor(images=img, return_tensors="pt")
    obj_inputs = obj_inputs.to(device)
    outputs = obj_model(**obj_inputs)
    target_sizes = torch.tensor([img.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    draw = ImageDraw.Draw(img, 'RGBA')

    color = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(obj_model.config.id2label))]

    for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
        box = [round(i, 2) for i in box.tolist()]
        label_name = obj_model.config.id2label[label.item()]
        outline = tuple(color[label])
        outline += (255, )
        fill = tuple(color[label])
        fill += (50, )
        fontsize = max(round(max(img.size) / 40), 12)
        # font = ImageFont.truetype("/worksapce/Sangwha/src/NanumGothic.ttf", fontsize)
        font = ImageFont.truetype("/app/src/font/NanumGothic.ttf", fontsize)
        txt_width, txt_height = font.getsize(label_name)
        
        draw.rectangle((box), outline=outline, fill=fill, width = 3)
        draw.text((box[0], box[1] - txt_height + 1), label_name, fill=(255, 255, 255), font=font)
        
    return img, cls_text
        # print(f"Classification : {cls_model.config.id2label[predicted_label]}")
        # print("Classification Time : {:.4f}sec".format((time.time() - cls_start_time)))

if __name__ == "__main__":
    if torch.cuda.is_available(): device = 'cuda'
    else: device ='cpu'
    
    # cls
    feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-xlarge-384-22k-1k")
    cls_model = ConvNextForImageClassification.from_pretrained("facebook/convnext-xlarge-384-22k-1k")
    cls_model = cls_model.to(device)
    
    # obj 
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
    obj_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
    obj_model = obj_model.to(device)


    demo = gr.Interface(predict, 
                        gr.Image(),
                        [
                            gr.Image(),
                            'text'
                        ],
                        title='Model Test',

                        )
    demo.launch() 





