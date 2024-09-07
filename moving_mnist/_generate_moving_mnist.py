import numpy as np
import argparse
import sys
import os
import torchvision.datasets as datasets
from torch.utils.data import Subset
from torchvision import transforms


class ToNumpy:
    def __init__(self):
        pass
    
    def __call__(self, im):
        return im.numpy()

def create_folder(path, verbose=False):
    # Do not create folder if folder exists
    if os.path.exists(path):
        if verbose:
            print(f"Folder {path} already exists.")
        return False
    # Create folder if folder does not exist
    os.makedirs(path)
    if verbose:
        print(f"Folder {path} created.")
    return True


def get_mnist(path, train=True, download=False, verbose=False):
    create_folder(path, verbose=verbose)
    transform=transforms.Compose([transforms.ToTensor(),
                                  ToNumpy()])
    mnist = datasets.MNIST(root=path, train=train,
                           download=download, transform=transform)
    if verbose:
        print(f'MNIST data is downloaded to {path}')
    return mnist


def filter_mnist_by_labels(mnist, labels):
    indices = np.where(np.isin(mnist.targets, labels))[0]
    return Subset(mnist, indices)


def convert_to_coco_bounding_boxes(bounding_boxes):
    bboxes = np.array(bounding_boxes)
    coco_bboxes = np.zeros_like(bboxes, dtype=np.int32)
    coco_bboxes[:,(0,1)] = bboxes[:,(0,1)]
    coco_bboxes[:,(2,3)] = bboxes[:,(2,3)] - bboxes[:,(0,1)]
    return coco_bboxes


def convert_to_yolo_bounding_boxes(bounding_boxes, width, height):
    bboxes = np.array(bounding_boxes)
    yolo_bboxes = np.zeros_like(bboxes, dtype=np.float32)
    coco_bboxes = convert_to_coco_bounding_boxes(bboxes)
    yolo_bboxes[:,(0,1)] = coco_bboxes[:,(0,1)] + coco_bboxes[:,(2,3)] / 2.0
    yolo_bboxes[:,(2,3)] = coco_bboxes[:,(2,3)]
    return normalize_bounding_boxes(yolo_bboxes, width, height)


def normalize_bounding_boxes(bounding_boxes, width, height):
    bboxes = np.array(bounding_boxes)
    norm_bboxes = np.zeros_like(bboxes, dtype=np.float32)
    norm_bboxes[:,(0,2)] = bboxes[:,(0,2)] / width
    norm_bboxes[:,(1,3)] = bboxes[:,(1,3)] / height
    return norm_bboxes


def paste_image(large_image, small_image, position):
    x, y= position
    # Get dimensions of large and small images
    _, large_h, large_w = large_image.shape
    _, small_h, small_w = small_image.shape
    
    # Calculate start and end points for small image on the large image
    x_start = max(x, 0)
    y_start = max(y, 0)
    x_end = min(x + small_w, large_w)
    y_end = min(y + small_h, large_h)
    
    # Calculate corresponding start and end points on the small image
    small_x_start = max(0, -x)
    small_y_start = max(0, -y)
    small_x_end = small_x_start + (x_end - x_start)
    small_y_end = small_y_start + (y_end - y_start)
    
    # Sum the relevant part of the small image with the large image
    large_image[:, y_start:y_end, x_start:x_end] += small_image[:, small_y_start:small_y_end, small_x_start:small_x_end]
    
    return large_image


def fit_bounding_box(frame):
    x_axis = np.nonzero(frame[0].sum(axis=0) > 1e-10)[0]
    y_axis = np.nonzero(frame[0].sum(axis=1) > 1e-10)[0]
    xmin, xmax = x_axis[0], x_axis[-1]
    ymin, ymax = y_axis[0], y_axis[-1]
    return xmin, ymin, xmax, ymax




def generate_moving_mnist(mnist, frame_shape, num_frames, num_sequences, nums_per_frame):
    # Get how many pixels can we move around a single image
    width, height = frame_shape
    original_size = mnist[0][0].shape[1]
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    sequences = np.empty((num_sequences, num_frames, 1, width, height), dtype=np.uint8)
    annotations = np.empty((num_sequences, num_frames, nums_per_frame, 5), dtype=np.int32)

    for sequence_idx in range(num_sequences):
        # Randomly generate direction, speed and velocity for both images
        direcs = np.pi * (np.random.rand(nums_per_frame) * 2 - 1)
        speeds = np.random.randint(5, size=nums_per_frame) + 2
        veloc = np.asarray([(speed * np.cos(direc), speed * np.sin(direc)) for direc, speed in zip(direcs, speeds)])
        # Get a list containing two PIL images randomly sampled from the database
        mnist_images = []
        mnist_labels = []
        for index in range(nums_per_frame):
            r = np.random.randint(0, len(mnist))
            mnist_images.append(mnist[r][0])
            mnist_labels.append(mnist[r][1])

        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(nums_per_frame)])

        # Generate new frames for the entire num_framesgth
        for frame_idx in range(num_frames):
            # canvases = [Image.new('L', (width, height)) for _ in range(nums_per_image)]
            canvases = [np.zeros((1, width, height), dtype=np.float32) for _ in range(nums_per_frame)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            # In canv (i.e Image object) place the image at the respective positions
            # Super impose both images on the canvas (i.e empty np array)
            for number_idx, canv in enumerate(canvases):
                # print(positions[number_idx])
                canv = paste_image(canv, mnist_images[number_idx], positions[number_idx].astype(np.int32))
                # canv.paste(mnist_images[i], tuple(positions[i].astype(int)))
                annotations[sequence_idx, frame_idx, number_idx, 0] = mnist_labels[number_idx]
                annotations[sequence_idx, frame_idx, number_idx, 1:] = fit_bounding_box(canv)
                canvas += canv
                # arr_from_img(canv, mean=0)

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

            # Make the permanent change to position by adding updated velocity
            positions = positions + veloc
            # Add the canvas to the dataset array
            sequences[sequence_idx, frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)
    return sequences, annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--dest', type=str, dest='dest', default='./data/processed/MovingMNIST')
    parser.add_argument('--frame_size', type=int, dest='frame_size', default=64)
    parser.add_argument('--num_frames', type=int, dest='num_frames', default=20)
    parser.add_argument('--num_sequences', type=int, dest='num_sequences', default=10000)
    parser.add_argument('--nums_per_frame', type=int, dest='nums_per_frame', default=2)
    args = parser.parse_args(sys.argv[1:])

    mnist_download_path = './data/raw'
    mnist = get_mnist(mnist_download_path, train=True,
                            download=True, verbose=True)
    filtered_mnist = filter_mnist_by_labels(mnist, [0, 1, 2, 3])
    frame_shape = (args.frame_size, args.frame_size)
    moving_mnist = generate_moving_mnist(filtered_mnist,
                                         frame_shape=frame_shape,
                                         num_frames=args.num_frames,
                                         num_sequences=args.num_sequences,
                                         nums_per_frame=args.nums_per_frame)

    create_folder(args.dest)
    sequences_path = os.path.join(args.dest, 'train-sequences.npy')
    annotations_path = os.path.join(args.dest, 'train-annotations.npy')
    np.save(sequences_path, moving_mnist[0])
    np.save(annotations_path, moving_mnist[1])
