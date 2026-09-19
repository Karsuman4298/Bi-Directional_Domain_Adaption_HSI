import os
import random
import numpy as np
import scipy.io as io
import tifffile as tiff
import h5py
import sklearn.model_selection
from sklearn import preprocessing
import torch
import torch.utils.data

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        try:
            # Try loading standard v7 and older MATLAB files
            return io.loadmat(dataset)
        except NotImplementedError:
            # Handle modern MATLAB v7.3 HDF5 files cleanly
            print("Loading MATLAB v7.3 file using h5py: {}".format(dataset))
            file = h5py.File(dataset, 'r')
            # MATLAB v7.3 dimensions are transposed relative to SciPy standard
            return {key: np.array(value).T for key, value in file.items()}
    elif ext == '.tif' or ext == '.tiff':
        return tiff.imread(dataset)
    else:
        raise ValueError("Unknown file format: {}".format(ext))
    
def load_mat_hsi(dataset_name, dataset_dir, norm='normband'):
    """ load HSI.mat dataset """
    available_sets = [
        'sa', 'pu', 'whulk', 'hrl', 'Loukia', 'Dioni', 'Houston18', 'Houston13',
    ]
    assert dataset_name in available_sets, "dataset should be one of " + str(available_sets)

    img = None
    gt = None
    labels = None

    if (dataset_name == 'sa'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_corrected.mat"))
        img = image['salinas_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_gt.mat"))
        gt = gt['salinas_gt']
        labels = [
            "Undefined", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow",
            "Fallow_rough_plow", "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained",
            "Soil_vinyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
            "Vinyard_untrained", "Vinyard_vertical_trellis",
        ]

    elif (dataset_name == 'pu'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU.mat"))
        img = image['paviaU']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU_gt.mat"))
        gt = gt['paviaU_gt']
        labels = [
            "Undefined", "Asphalt", "Meadows", "Gravel", "Trees", 
            "Painted metal sheets", "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows",
        ]

    elif (dataset_name == 'whulk'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou.mat"))
        img = image['WHU_Hi_LongKou']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou_gt.mat"))
        gt = gt['WHU_Hi_LongKou_gt']
        labels = [
            'Undefined', 'Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',
            'Narrow-leaf soybean', 'Rice', 'Water', 'Roads and houses', 'Mixed weed',
        ]

    elif (dataset_name == 'Loukia'):
        img = open_file(dataset_dir + 'Loukia.mat')['ori_data']
        gt = open_file(dataset_dir + 'Loukia_gt_out68.mat')['map']
        labels = [
            'Undefined', 'Dense Urban Fabric', 'Mineral Extraction Sites', 'Non Irrigated Arable Land',
            'Fruit Trees', 'Olive Groves', 'Coniferous Forest', 'Dense Sclerophyllous Vegetation',
            'Sparce Sclerophyllous Vegetation', 'Sparcely Vegetated Areas', 'Rocks and Sand', 'Water', 'Coastal Water'
        ]

    elif dataset_name == 'Dioni':
        img = open_file(dataset_dir + 'Dioni.mat')['ori_data']
        gt = open_file(dataset_dir + 'Dioni_gt_out68.mat')['map']
        labels = [
            'Undefined', 'Dense Urban Fabric', 'Mineral Extraction Sites', 'Non Irrigated Arable Land',
            'Fruit Trees', 'Olive Groves', 'Coniferous Forest', 'Dense Sclerophyllous Vegetation',
            'Sparce Sclerophyllous Vegetation', 'Sparcely Vegetated Areas', 'Rocks and Sand', 'Water', 'Coastal Water'
        ]

    elif dataset_name == 'Houston18':
        import h5py
        with h5py.File(dataset_dir + 'Houston18.mat', 'r') as f:
            img = np.array(f['ori_data']).T
        with h5py.File(dataset_dir + 'Houston18_7gt.mat', 'r') as f:
            gt = np.array(f['map']).T.astype(int)
        labels = ['0', "1", "2", "3", "4", "5", "6", "7"]

    elif dataset_name == 'Houston13':
        import h5py
        with h5py.File(dataset_dir + 'Houston13.mat', 'r') as f:
            img = np.array(f['ori_data']).T
        with h5py.File(dataset_dir + 'Houston13_7gt.mat', 'r') as f:
            gt = np.array(f['map']).T.astype(int)
        labels = ['0', "1", "2", "3", "4", "5", "6", "7"]

    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("warning: nan values found in dataset {}, using 0 replace them".format(dataset_name))
        img[nan_mask] = 0
        gt[nan_mask] = 0

    if norm == 'normband':
        img = np.asarray(img, dtype='float32')
        m, n, d = img.shape[0], img.shape[1], img.shape[2]
        img_ori = img.reshape((m*n,-1))
        index = np.where(img_ori.sum(axis=-1)!=0)
        img = img_ori[index]
        img = img/img.max()
        img_temp = np.sqrt(np.asarray((img**2).sum(1)))
        img_temp = np.expand_dims(img_temp,axis=1)
        img_temp = img_temp.repeat(d,axis=1)
        img_temp[img_temp==0]=1
        img = img/img_temp
        img_ori[index] = img
        img = np.reshape(img_ori,(m,n,-1))
    elif norm == 'minmax':
        img = np.asarray(img, dtype=np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        mean_by_c = np.mean(img, axis=(0, 1))
        for c in range(img.shape[-1]):
            img[:, :, c] = img[:, :, c] - mean_by_c[c]
    elif norm == 'std':
        meanhsi = np.mean(np.reshape(img, -1))
        sigmahsi = np.sqrt(np.var(np.reshape(img, -1)))
        img = (img - meanhsi) / (sigmahsi)
    elif norm == 'ln':
        img =  img/(np.sqrt(np.sum(img**2,axis=2,keepdims = True))+ 1e-9)
    elif norm =='sklearn':
        data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))
        data_scaler = preprocessing.scale(data)
        img = data_scaler.reshape(img.shape[0], img.shape[1], img.shape[2])
    elif norm =='ori':
        pass

    gt = gt.astype('int') - 1
    labels = labels[1:]

    return img, gt, labels

def sample_gt(gt, train_size, seed, mode='random'):
    """Extract a fixed percentage of samples from an array of labels."""
    indices = np.where(gt >= 0)
    X = list(zip(*indices))
    y = gt[indices].ravel()
    train_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)
    if train_size > 1:
       train_size = int(train_size)
    train_label = []
    test_label = []
    if mode == 'random':
        if train_size == 1:
            random.seed(seed)
            random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(
                X, train_size=train_size, stratify=y, random_state=seed
            )
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
            
    return train_gt, test_gt
class HSIDataset(torch.utils.data.Dataset):
  
    """ Custom PyTorch Dataset wrapping patch extraction for Hyperspectral images """
    def __init__(self, data, gt, patch_size=13, transform=None, data_aug=False, **kwargs):
        self.data = data
        self.gt = gt
        self.patch_size = patch_size
        self.transform = transform
        self.data_aug = data_aug  # Captures the data_aug flag cleanly
        
        # Collect background and labeled foreground pixels cleanly
        self.indices = np.argwhere(self.gt >= 0)
        
        # Reflect padding configuration for clean patches near edges
        margin = int((self.patch_size - 1) / 2)
        self.padded_data = np.pad(
            self.data, 
            ((margin, margin), (margin, margin), (0, 0)), 
            mode='reflect'
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        r, c = self.indices[idx]
        margin = int((self.patch_size - 1) / 2)
        
        # Window slicing logic around coordinate point
        patch = self.padded_data[r : r + self.patch_size, c : c + self.patch_size, :]
        label = self.gt[r, c]
        
        # Basic on-the-fly data augmentation if enabled and in training mode
        if self.data_aug and random.random() > 0.5:
            # Randomly flip patch vertically or horizontally
            if random.random() > 0.5:
                patch = np.flip(patch, axis=0)
            else:
                patch = np.flip(patch, axis=1)
        
        # Channel transpose from (H, W, C) to deep learning format (C, H, W)
        patch = torch.FloatTensor(patch.transpose(2, 0, 1).copy())
        patch = patch.unsqueeze(0)
        label = torch.LongTensor([label]).squeeze()
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch, label


# ─── TSTnet-compatible API ───────────────────────────────────────────────────

HOUSTON_PALETTE = {
    0: (0, 0, 0),       # Background
    1: (255, 0, 0),     # Class 1
    2: (0, 255, 0),     # Class 2
    3: (0, 0, 255),     # Class 3
    4: (255, 255, 0),   # Class 4
    5: (255, 0, 255),   # Class 5
    6: (0, 255, 255),   # Class 6
    7: (128, 128, 0),   # Class 7
}

def get_dataset(dataset_name, dataset_dir, norm='normband'):
    """Load a dataset and return (img, gt, label_values, ignored_labels, rgb_bands, palette).
    
    This wraps load_mat_hsi to match the 6-return signature expected by TSTnet.
    Note: load_mat_hsi returns gt 0-indexed (subtracts 1), but TSTnet expects
    1-indexed gt (0 = background). We add 1 back to restore original convention.
    """
    img, gt, labels = load_mat_hsi(dataset_name, dataset_dir, norm=norm)
    # load_mat_hsi subtracts 1 from gt; TSTnet expects original 1-indexed labels
    gt = gt + 1
    label_values = ['Undefined'] + labels
    ignored_labels = [0]
    rgb_bands = (0, 1, 2)
    palette = HOUSTON_PALETTE
    return img, gt, label_values, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    """TSTnet-compatible dataset that produces (1, C, P, P) patches with 1-indexed labels."""

    def __init__(self, data, gt, patch_size=13, flip_augmentation=False,
                 radiation_augmentation=False, mixture_augmentation=False,
                 center_pixel=False, supervision='full', **kwargs):
        super().__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.center_pixel = center_pixel
        self.supervision = supervision

        # Build index of labeled pixels
        mask = np.ones_like(gt, dtype=bool)
        for l in kwargs.get('ignored_labels', [0]):
            mask[gt == l] = False
        self.indices = np.argwhere(mask)

        # Reflect-pad for border patches
        margin = patch_size // 2
        self.padded_data = np.pad(
            data,
            ((margin, margin), (margin, margin), (0, 0)),
            mode='reflect'
        )

        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.indices[idx]
        margin = self.patch_size // 2
        patch = self.padded_data[x:x + self.patch_size, y:y + self.patch_size, :]
        label = self.label[x, y]

        # Data augmentation
        if self.flip_augmentation and np.random.random() > 0.5:
            patch = np.flip(patch, axis=np.random.randint(2)).copy()
        if self.radiation_augmentation and np.random.random() > 0.5:
            patch = patch * (1 + np.random.uniform(-0.1, 0.1))

        # (H, W, C) -> (1, C, H, W)
        patch = np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32)
        patch = torch.from_numpy(patch).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return patch, label


class data_prefetcher:
    """Async CUDA data prefetcher — wraps a DataLoader for GPU-side pre-loading."""

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.cuda(non_blocking=True)
                self.next_target = self.next_target.cuda(non_blocking=True)

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is None:
            raise StopIteration
        self.preload()
        return input, target

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

