import datasets
import pytorch_lightning
import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms
import torchvision.transforms.v2
import transformers
from typing import *


class ImageClassificationDataCollator():
    """
    Utility function object that manages the creation of batches for
    training, validating and testing PyTorch models for image
    classification.
    """
    
    def __init__(self) -> None:
        """
        Constructor.
        """

        pass

    def __call__(
            self,
            data_points: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Functional call. Create a batch from a list of data points.

        Args:
            data_points (List[Dict[str, torch.Tensor]]): The list of data points
            to batch.

        Returns:
            Dict[str, torch.Tensor]: The created batch.
        """

        # Create auxiliary lists.
        pixel_values = list()
        labels = list()

        # Iterate on the list of data point to populate the auxiliary lists.
        for data_point in data_points:
            pixel_values.append(data_point["pixel_values"])
            labels.append(data_point["labels"])

        # Create the batch as a dict().
        return {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }


class ImageClassificationDataModule(pytorch_lightning.LightningDataModule):
    """
    A PyTorch Lightning data module that handles data for a simple
    image classification task that will use a DeiT transformer or a
    variant of it.
    """

    def __init__(
            self,
            hf_model_name: str,
            batch_size: int
    ) -> None:
        """
        Constructor.

        Args:
            hf_model_name (str): The full HuggingFace name of the DeiT
            transformer model that will be used in the task, like
            'facebook/deit-tiny-distilled-patch16-224'.

            batch_size (int): The batch size to use for the underlying
            PyTorch datasets and dataloaders.
        """

        super(ImageClassificationDataModule, self).__init__()

        self.hf_model_name = hf_model_name
        self.batch_size = batch_size

        # Prepare image augmentations like the ones performed on the
        # ImageNet dataset.
        self.img_transforms = torchvision.transforms.v2.AutoAugment(
            policy=torchvision.transforms.v2.AutoAugmentPolicy.IMAGENET
        )

        # Prepare the data module state to populate after.
        self.img_processor = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self) -> None:
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Download the image preprocessor built in the chosen
        # HuggingFace DeiT transformer model and save it in
        # a local directory.
        transformers.DeiTImageProcessor.from_pretrained(
            self.hf_model_name,
            cache_dir="./hf/img_processors"
        )

    def setup(self, stage: str) -> None:
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        def preprocess(batch) -> Dict[str, torch.Tensor]:
            """
            Auxiliary function that preprocess a batch of data
            (for performances).

            Args:
                batch (...): The batch to preprocess.

            Returns:
                Dict[str, torch.Tensor]: The preprocessed batch.
            """

            # If the model is fitting (i.e. training AND validating), perform
            # image augmentation and then preprocess the images.
            # Then transform the images into PyTorch tensors.
            if stage == "fit":
                results = self.img_processor(
                    list(
                        map(
                            lambda img: self.img_transforms(img.convert("RGB")),
                            batch["image"]
                        )
                    ),
                    return_tensors="pt"
                )
            # Otherwise if the model is not fitting (i.e. validating OR testing),
            # only preprocess the images.
            # Then transform the images into PyTorch tensors.
            else:
                results = self.img_processor(
                    list(
                        map(
                            lambda img: img.convert("RGB"), batch["image"]
                        )
                    ),
                    return_tensors="pt"
                )
            
            # Create the preprocessed batch transforming all remaining data
            # into PyTorch tensors.
            return {
                "pixel_values": results["pixel_values"],
                "labels": torch.tensor(batch["labels"], dtype=torch.int64)
            }

        
        # Based on the fact that the model is fitting, validating or testing,
        # load the image preprocessor from the local directory and perform the
        # following operations on the right datasets:
        # 1. set the preprocessing as a transformation on the dataset to be
        # applied right before the actual data is returned.
        if stage == "fit":
            self.img_processor = transformers.DeiTImageProcessor.from_pretrained(
                self.hf_model_name,
                cache_dir="./hf/img_processors"
            )

            self.train_ds.set_transform(preprocess, columns=["image", "labels"])
            self.val_ds.set_transform(preprocess, columns=["image", "labels"])

        elif stage == "validate":
            self.img_processor = transformers.DeiTImageProcessor.from_pretrained(
                self.hf_model_name,
                cache_dir="./hf/img_processors"
            )

            self.val_ds.set_transform(preprocess, columns=["image", "labels"])

        elif stage == "test":
            self.img_processor = transformers.DeiTImageProcessor.from_pretrained(
                self.hf_model_name,
                cache_dir="./hf/img_processors"
            )

            self.test_ds.set_transform(preprocess, columns=["image", "labels"])

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Create a PyTorch dataloader for the training dataset.
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Create a PyTorch dataloader for the validation dataset.
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Create a PyTorch dataloader for the testing dataset.
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False
        )


class Caltech256DataModule(ImageClassificationDataModule):
    """
    A PyTorch Lightning data module that handles the data of the
    Caltech-256 dataset (downloader from HuggingFace with the full name
    'ilee0022/Caltech-256').
    """

    def __init__(
            self,
            hf_model_name: str,
            batch_size: int
    ) -> None:
        """
        Constructor.

        Args:
            hf_model_name (str): The full HuggingFace name of the DeiT
            transformer model that will be used in the task, like
            'facebook/deit-tiny-distilled-patch16-224'.

            batch_size (int): The batch size to use for the underlying
            PyTorch datasets and dataloaders.
        """

        super(Caltech256DataModule, self).__init__(
            hf_model_name, batch_size
        )

    def prepare_data(self) -> None:
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        super(Caltech256DataModule, self).prepare_data()

        # Download the training set of the Caltech-256 dataset
        # from HuggingFace and save it in a local directory.
        datasets.load_dataset(
            "ilee0022/Caltech-256", cache_dir="./hf/datasets",
            split="train", trust_remote_code=True
        )

        # Download the validation set of the Caltech-256 dataset
        # from HuggingFace and save it in a local directory.
        datasets.load_dataset(
            "ilee0022/Caltech-256", cache_dir="./hf/datasets",
            split="validation", trust_remote_code=True
        )

        # Download the test set of the Caltech-256 dataset
        # from HuggingFace and save it in a local directory.
        datasets.load_dataset(
            "ilee0022/Caltech-256", cache_dir="./hf/datasets",
            split="test", trust_remote_code=True
        )

    def setup(self, stage: str) -> None:
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Based on the fact that the model is fitting, validating or testing,
        # load the right set of the Caltech-256 dataset from the local
        # directory and perform the
        # following operations on the right sets:
        # 1. remove the column 'text'.
        # 2. rename the column 'label' into 'labels'.
        # 3. subtract 1 to each label in order to be from the range [1, 257] to
        # the better range [0, 256] for a PyTorch model.
        if stage == "fit":
            self.train_ds = datasets.load_dataset(
                "ilee0022/Caltech-256", cache_dir="./hf/datasets",
                split="train", trust_remote_code=True
            )
            self.train_ds = self.train_ds.remove_columns("text")
            self.train_ds = self.train_ds.rename_column("label", "labels")
            self.train_ds = self.train_ds.map(
                function=lambda batch: {"labels": list(map(lambda label: label - 1, batch["labels"]))},
                batched=True
            )
            self.val_ds = datasets.load_dataset(
                "ilee0022/Caltech-256", cache_dir="./hf/datasets",
                split="validation", trust_remote_code=True
            )
            self.val_ds = self.val_ds.remove_columns("text")
            self.val_ds = self.val_ds.rename_column("label", "labels")
            self.val_ds = self.val_ds.map(
                function=lambda batch: {"labels": list(map(lambda label: label - 1, batch["labels"]))},
                batched=True
            )

        elif stage == "validate":
            self.val_ds = datasets.load_dataset(
                "ilee0022/Caltech-256", cache_dir="./hf/datasets",
                split="validation", trust_remote_code=True
            )
            self.val_ds = self.val_ds.remove_columns("text")
            self.val_ds = self.val_ds.rename_column("label", "labels")
            self.val_ds = self.val_ds.map(
                function=lambda batch: {"labels": list(map(lambda label: label - 1, batch["labels"]))},
                batched=True
            )

        elif stage == "test":
            self.test_ds = datasets.load_dataset(
                "ilee0022/Caltech-256", cache_dir="./hf/datasets",
                split="test", trust_remote_code=True
            )
            self.test_ds = self.test_ds.remove_columns("text")
            self.test_ds = self.test_ds.rename_column("label", "labels")
            self.test_ds = self.test_ds.map(
                function=lambda batch: {"labels": list(map(lambda label: label - 1, batch["labels"]))},
                batched=True
            )

        super(Caltech256DataModule, self).setup(stage)
