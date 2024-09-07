from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from moving_mnist import MovingMNIST
from matplotlib import pyplot as plt
import os


root = './data/processed/MovingMNIST'
num_sequences = 2
num_frame = 4


dataset = MovingMNIST(root=root, download=False)
mnist = DataLoader(dataset, batch_size=num_sequences, shuffle=True)


sequences, annotations = next(iter(mnist))
fig, axs = plt.subplots(num_sequences, num_frame)
for seq_idx in range(num_sequences):
    for frame_idx in range(num_frame):
        frame = sequences[seq_idx, frame_idx]
        boxes = annotations[seq_idx, frame_idx, :, 1:]
        labels = annotations[seq_idx, frame_idx, :, 0]
        labels = [str(i.item()) for i in labels]
        draw = draw_bounding_boxes(frame, boxes, labels=labels,
                                   colors='red', width=1)
        axs[seq_idx, frame_idx].imshow(F.to_pil_image(draw))

plt.savefig(os.path.join(root, 'output.png'))
