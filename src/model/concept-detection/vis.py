from transformers import AutoModel,AutoImageProcessor
import cv2
import matplotlib.pyplot as plt
import torch
from prepare_data import train_archive
import numpy as np
from PIL import Image

def vis(IMG_MODEL):
    def plot_attention_map(original_img, att_map):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map Last Layer')
        _ = ax1.imshow(original_img)
        _ = ax2.imshow(att_map)
        plt.show()

    #IMG_MODEL = 'microsoft/beit-base-patch16-224-pt22k-ft22k' 

    preprocessor = AutoImageProcessor.from_pretrained(IMG_MODEL)
    image_encoder = AutoModel.from_pretrained(IMG_MODEL)
    #valid/ImageCLEFmedical_Caption_2024_train_017527.jpg
    img1 = Image.open(train_archive.open(f'train/ImageCLEFmedical_Caption_2024_train_017527.jpg')).convert('RGB')

    img_prep = preprocessor(images=[img1],return_tensors="pt")['pixel_values']
    out = image_encoder(pixel_values=img_prep,return_dict=True,output_attentions=True)

    att_mat = torch.mean(out['attentions'][-1], dim=1)

    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    result = cv2.resize(mask / mask.max(), img1.size)
    plot_attention_map(img1,result)