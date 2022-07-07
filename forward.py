import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

k = 10

imagenet_labels = dict(enumerate(open('classes')))

model = torch.load('model.pth')
model.eval()

img = Image.open('/home/angepapa/Desktop/papadopoulos corosect/Datasets/after_spain_dataset/cf2_files/captures/crickets/63/real/center/att_2/frame_55.png')
img = img.resize((384, 384))
img = (np.array(img) / 128) - 1                             # in the range -1, 1
input = torch.from_numpy(img).permute(2, 0, 1)
input = input.unsqueeze(0).to(torch.float32)                                 # make it like a batch [1, 3, 384, 384]

logits = model(input)

probs = F.softmax(logits, dim=-1)

top_probs, top_ixs = probs[0].topk(k)

for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix].strip()
    print(f'{i}: {cls:<60} --- {prob:.4f}')

