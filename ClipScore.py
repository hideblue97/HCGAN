import os
from PIL import Image
import numpy as np
import torch
import clip
import matplotlib.pyplot as plt

#
model, preprocess = clip.load("ViT-B/32")
# print(model)
model.cuda().eval()

#
your_image_folder = "put your image folder here"
your_texts = ["forest with smoke and haze"]

images = []
for filename in os.listdir(your_image_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        path = os.path.join(your_image_folder, filename)
        image = Image.open(path).convert("RGB")
        images.append(preprocess(image))


image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize(your_texts).cuda()


with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()


image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarity = (text_features.cpu().numpy() @ image_features.cpu().numpy().T)
print('similarity:',similarity)


'''plt.imshow(similarity, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.xlabel("Images")
plt.ylabel("Texts")
plt.title("Similarity between Texts and Images")
plt.xticks(range(len(images)), range(len(images)), rotation=90)
plt.yticks(range(len(your_texts)), your_texts, rotation='vertical', va='center') 


plt.savefig("similarity.png", bbox_inches='tight') 


plt.show()'''
