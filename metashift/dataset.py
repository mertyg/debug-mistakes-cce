import pickle
import numpy as np
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from constants import METASHIFT_IMAGES, CANDIDATES


class ListDataset:
    def __init__(self, images, class_to_idx=None, labels=-1, preprocess=None):
        self.images = images
        self.preprocess = preprocess
        self.labels = labels
        self.class_to_idx = class_to_idx

    def __len__(self):
        # Return the length of the dataset
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels if isinstance(self.labels, int) else self.labels[idx]
        if self.preprocess:
            image = self.preprocess(image)
        return image, label


class MetashiftManager(object):
    def __init__(self, base_dir=METASHIFT_IMAGES, seed=1):
        self.candidates = {}
        cand_dict = pickle.load(open(CANDIDATES, "rb"))
        for key, ims in cand_dict.items():
            self.candidates[key] = [f"{base_dir}{path}.jpg" for path in ims]
        self.base_dir = base_dir
        self.seed = seed
  
    def get_single_class(self, cls, n_samples=None, exclude=None):
        template = f"{cls}("
        all_keys = [subset for subset in self.candidates.keys() if template in subset]
        cls_images = set()
        for subset in all_keys:
            cls_images |= set(self.candidates[subset])
        cls_images = list(cls_images)
        if exclude is not None:
            exclude_ims = self.get_subset_ims(exclude)
            cls_images = np.setdiff1d(cls_images, exclude_ims)
        if n_samples is None:
            return cls_images
        else:
            return np.random.choice(cls_images, n_samples, replace=False)
    
    def get_class_ims(self, classes: list, n_samples=None):
        class_images = {}
        for cls in classes:
            class_images[cls] = self.get_single_class(cls, n_samples)
        return class_images

    def get_subset_ims(self, subset: str, n_samples:int = None, exclude:str = None):
        ims = self.candidates[subset]
        if exclude is not None:
            exclude_ims = self.get_subset_ims(exclude)
            ims = np.setdiff1d(ims, exclude_ims)
        if n_samples is None:
            return ims
        #n_samples = min(n_samples, len(ims))
        return np.random.choice(ims, n_samples, replace=True)


def build_dataset(args, classes, train_domain, 
                  n_train_per_class=50, n_val_per_class=50, n_test_per_class=50):
    """
    Args:
        args:
        classes (list[str]): A list of class names 
        train_domain (str): The domain to use for training. e.g. dog(snow) for dog images with snow in the background

    """
    shift_class = train_domain.split("(")[0]
    manager = MetashiftManager(seed=args.seed)
    data_meta_info = {"train": {"ims": [], "lbls": []}, 
                      "val": {"ims": [], "lbls": []}, 
                      "test": {"ims": [], "lbls": []}}
    cls_to_lbl = {}
    
    for i, c in enumerate(classes):
        cls_to_lbl[c] = i
        data_meta_info[c] = {}
        
        if c != shift_class:
            cls_images = manager.get_single_class(c, n_samples=n_train_per_class+n_val_per_class+n_test_per_class)
            data_meta_info[c]["train"] = cls_images[:n_train_per_class]
            data_meta_info[c]["val"] = cls_images[n_train_per_class:n_train_per_class+n_val_per_class]
            data_meta_info[c]["test"] = cls_images[-n_test_per_class:]
        
        else:
            train_domain_ims = manager.get_subset_ims(train_domain, n_samples=n_train_per_class)
            test_domain_ims = manager.get_single_class(c, n_samples=n_val_per_class+n_test_per_class, exclude=train_domain)

            data_meta_info[c]["train"] = train_domain_ims
            data_meta_info[c]["val"] = test_domain_ims[:n_val_per_class]
            data_meta_info[c]["test"] = test_domain_ims[n_val_per_class:]
        
        data_meta_info["train"]["ims"].extend(data_meta_info[c]["train"])
        data_meta_info["train"]["lbls"].extend([i for _ in range(n_train_per_class)])
        
        data_meta_info["val"]["ims"].extend(data_meta_info[c]["val"])
        data_meta_info["val"]["lbls"].extend([i for _ in range(n_val_per_class)]) 
           
        data_meta_info["test"]["ims"].extend(data_meta_info[c]["test"])
        data_meta_info["test"]["lbls"].extend([i for _ in range(n_test_per_class)])
                    
    return data_meta_info, cls_to_lbl


def load_data(args, train_preprocess, val_preprocess, dataset):
    # expect something of form 
    classes, train_domain = dataset.split(":")
    classes = classes.split("-")
    data_meta_info, cls_to_lbl = build_dataset(args, classes, train_domain)
    loaders = {}
    train_ds = ListDataset(data_meta_info["train"]["ims"], class_to_idx=cls_to_lbl, labels=data_meta_info["train"]["lbls"], preprocess=train_preprocess)
    loaders['train'] = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    val_ds = ListDataset(data_meta_info["val"]["ims"], class_to_idx=cls_to_lbl, labels=data_meta_info["val"]["lbls"], preprocess=val_preprocess)
    loaders['val'] = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    
    test_ds = ListDataset(data_meta_info["test"]["ims"], class_to_idx=cls_to_lbl, labels=data_meta_info["test"]["lbls"], preprocess=val_preprocess)
    loaders['test'] = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    
    print(f'Train, Val, Test: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}')
    return loaders, cls_to_lbl, data_meta_info
