from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from moving_mnist import MovingMNIST
import numpy as np
from PIL import Image
import imageio
import os


root = './data/processed/MovingMNIST/'
output = './images/'
num_sequences = 6
size = (128, 128)


dataset = MovingMNIST(root=root, download=False)
mnist = DataLoader(dataset, batch_size=num_sequences, shuffle=True)


sequences, annotations = next(iter(mnist))
num_frames = sequences.shape[1]
for seq_idx in range(num_sequences):
    image_seq = []
    for frame_idx in range(num_frames):
        frame = sequences[seq_idx, frame_idx]
        boxes = annotations[seq_idx, frame_idx, :, 1:]
        labels = annotations[seq_idx, frame_idx, :, 0]
        labels = [str(i.item()) for i in labels]
        draw = draw_bounding_boxes(frame, boxes, labels=labels,
                                   colors='red', width=1)
        image = draw.permute(1, 2, 0)
        image = Image.fromarray(image.numpy().astype(np.uint8))
        image = image.resize(size, Image.LANCZOS)
        image_seq.append(image)

    filename = 'seq' + str(seq_idx) + '.gif'
    path = os.path.join(output, filename)
    imageio.mimsave(path, image_seq, duration=0.1, loop=0)
