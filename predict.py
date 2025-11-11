# imports the model architecture
# loads the saved weights: Use torch.load function
# loads the test set of a DatasetLoader (see train.py)
# Iterate over the test set images, generate predictions, save segmentation masks

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import glob
from torch.utils.data import DataLoader
from lib.model.EncDecModel import EncDec
from lib.dataset.PhCDataset import PhC


def save_mask(array, path):
    # array should be a 2D numpy array with 0s and 1s
    # np.unique(array) == [0, 1]
    # len(np.shape(array)) == 2
    im_arr = (array*255)
    Image.fromarray(np.uint8(im_arr)).save(path)

class PhCTest(torch.utils.data.Dataset):
    def __init__(self, transform):
        'Initialization'
        self.transform = transform
        data_path = os.path.join('/dtu/datasets1/02516/phc_data', 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.jpg'))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        X = self.transform(image)
        return X, image_path

@torch.no_grad()
def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = EncDec().to(device)
    model.load_state_dict(torch.load("runs/model.pth", map_location=device))
    model.eval()

    # Dataset
    size = 128
    test_transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor()])
    testset = PhCTest(transform=test_transform)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False,
                             num_workers=3)

    # Create output directory
    os.makedirs("predicted_masks", exist_ok=True)

    # Inference loop
    for X, image_path in test_loader:
        X = X.to(device)
        Y_hat = torch.sigmoid(model(X)).detach().cpu().squeeze().numpy()
        Y_mask = (Y_hat >= 0.5).astype(np.uint8)

        # Save mask
        base_name = os.path.basename(image_path[0])
        mask_name = os.path.splitext(base_name)[0] + '_mask.png'
        save_path = os.path.join("predicted_masks", mask_name)
        save_mask(Y_mask, save_path)
        print(f"Saved mask to {save_path}")

if __name__ == "__main__":
    main()
