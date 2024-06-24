import pytorch_lightning
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.optim.lr_scheduler
import torchmetrics
import torchmetrics.functional
import torchmetrics.functional.classification
from typing import *
from .models import DeiTForImageClassification, AViTForImageClassification


class DeiTModel(pytorch_lightning.LightningModule):
    """
    A PyTorch Lightning model that wraps the DeiTForImageClassification
    PyTorch model (See documentation).
    """

    def __init__(self, avit_kwargs: Dict[str, Any], lr: float = 1.e-4) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.

            lr (float): The learning rate.
        """

        super(DeiTModel, self).__init__()

        # Save the hyperparameters on a YAML file.
        self.save_hyperparameters(ignore=["avit_kwargs"])

        # Retrieve the number of classes.
        self.num_classes: int = avit_kwargs["num_classes"]

        # The underlying PyTorch model.
        self.transf_model = DeiTForImageClassification(avit_kwargs)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Feedforward for this PyTorch Lightning model.
        In particular compute the logits for the classification from a batch of
        pixels from images.

        Args:
            pixel_values (torch.Tensor): The batch of pixels to start with.
            [batch_size, num_channels=3, height=224, width=224]

        Returns:
            torch.Tensor: The computed batch of final hidden states.
            [batch_size, num_classes=257]
        """

        # Compute the logits.
        logits = self.transf_model(pixel_values)
        # logits [batch_size, num_classes=257]

        return logits

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the (comprehensive) loss function for this PyTorch Lightning model.
        In particular, this is the cross-entropy loss function.

        Args:
            logits (torch.Tensor): The logits, i.e. prediction.
            [batch_size, num_classes=257]

            labels (torch.Tensor): The labels, i.e. ground truth.
            [batch_size]

        Returns:
            torch.Tensor: The computed loss.
            [0]
        """

        return torch.nn.functional.cross_entropy(logits, labels)

    def accuracy_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the accuracy metric for this PyTorch Lightning model.

        Args:
            logits (torch.Tensor): The logits, i.e. prediction.
            [batch_size, num_classes=257]

            labels (torch.Tensor): The labels, i.e. ground truth.
            [batch_size]

        Returns:
            torch.Tensor: The computed accuracy.
            [0]
        """

        return torchmetrics.functional.classification.multiclass_accuracy(
            logits, labels, self.num_classes, average="macro"
        )

    def f1_score_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the F1 score metric for this PyTorch Lightning model.

        Args:
            logits (torch.Tensor): The logits, i.e. prediction.
            [batch_size, num_classes=257]

            labels (torch.Tensor): The labels, i.e. ground truth.
            [batch_size]

        Returns:
            torch.Tensor: The computed F1 score.
            [0]
        """

        return torchmetrics.functional.classification.multiclass_f1_score(
            logits, labels, self.num_classes, average="macro"
        )

    def configure_optimizers(self):
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # The optimizer.
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            eps=1.e-8, weight_decay=0.01
        )

        # The learning rate scheduler.
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss_epoch",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Compute, log and return the desired losses and metrics during
        # training.

        labels = batch["labels"].view(-1)
        logits = self(batch["pixel_values"]).view(-1, self.num_classes)
        loss = self.loss_fn(logits, labels)
        accuracy = self.accuracy_fn(logits, labels)
        f1_score = self.f1_score_fn(logits, labels)

        self.log(
            "train_loss", loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "train_accuracy", accuracy,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "train_f1_score", f1_score,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "f1_score": f1_score
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Compute, log and return the desired losses and metrics during
        # validation.

        labels = batch["labels"].view(-1)
        logits = self(batch["pixel_values"]).view(-1, self.num_classes)
        loss = self.loss_fn(logits, labels)
        accuracy = self.accuracy_fn(logits, labels)
        f1_score = self.f1_score_fn(logits, labels)

        self.log(
            "val_loss", loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "val_accuracy", accuracy,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "val_f1_score", f1_score,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "f1_score": f1_score
        }

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Compute, log and return the desired losses and metrics during
        # testing.

        labels = batch["labels"].view(-1)
        logits = self(batch["pixel_values"]).view(-1, self.num_classes)
        loss = self.loss_fn(logits, labels)
        accuracy = self.accuracy_fn(logits, labels)
        f1_score = self.f1_score_fn(logits, labels)

        self.log(
            "test_loss", loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "test_accuracy", accuracy,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "test_f1_score", f1_score,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "f1_score": f1_score
        }


class AViTModel(DeiTModel):
    """
    A PyTorch Lightning model that wraps the AViTForImageClassification
    PyTorch model (See documentation).
    """

    def __init__(self, avit_kwargs: Dict[str, Any], lr: float = 1.e-4) -> None:
        super(AViTModel, self).__init__(avit_kwargs, lr=lr)

        # Retrieve the distribution loss and the ponder loss scaling factors.
        alpha_distr: float = avit_kwargs["alpha_distr"]
        alpha_ponder: float = avit_kwargs["alpha_ponder"]

        self.alpha_distr = alpha_distr
        self.alpha_ponder = alpha_ponder

        # The underlying PyTorch model.
        self.transf_model = AViTForImageClassification(avit_kwargs)

    def ponder_loss_fn(self, rhos: torch.Tensor) -> torch.Tensor:
        """
        Compute the ponder loss function for this PyTorch Lightning model.
        In particular, this is the average of the rhos (intermediate
        ponder losses at each AViT layer).

        Args:
            rhos (torch.Tensor): The rhos.
            [batch_size, seq_len=197]

        Returns:
            torch.Tensor: The computed loss.
            [0]
        """

        return torch.mean(rhos)

    def distr_loss_fn(
            self,
            halting_score_distr: torch.Tensor,
            halting_score_distr_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the distribution loss function for this PyTorch Lightning model.
        In particular, this is the Kullback-Leibler divergence loss between
        the halting score probability distribution and
        the halting score target probability distribution.

        Args:
            halting_score_distr (torch.Tensor): The halting score
            probability distribution.
            [depth=12]

            halting_score_distr_target (torch.Tensor): The halting score target
            probability distribution.
            [depth=12]

        Returns:
            torch.Tensor: The computed loss.
            [0]
        """

        return torch.nn.functional.kl_div(
            halting_score_distr.log(), halting_score_distr_target,
            reduction="batchmean", log_target=False
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Compute, log and return the desired losses and metrics during
        # training.

        labels = batch["labels"].view(-1)
        outputs = self(batch["pixel_values"])
        logits = outputs["logits"].view(-1, self.num_classes)
        task_loss = self.loss_fn(logits, labels)
        rhos = outputs["rhos"]
        ponder_loss = self.ponder_loss_fn(rhos)
        halting_score_distr = outputs["halting_score_distr"]
        halting_score_distr_target = outputs["halting_score_distr_target"]
        distr_loss = self.distr_loss_fn(
            halting_score_distr, halting_score_distr_target
        )
        loss = (
            task_loss
            + self.alpha_ponder * ponder_loss
            + self.alpha_distr * distr_loss
        )
        accuracy = self.accuracy_fn(logits, labels)
        f1_score = self.f1_score_fn(logits, labels)

        self.log(
            "train_loss", loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "train_task_loss", task_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "train_ponder_loss", ponder_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "train_distr_loss", distr_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "train_accuracy", accuracy,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "train_f1_score", f1_score,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )

        return {
            "loss": loss,
            "task_loss": task_loss,
            "ponder_loss": ponder_loss,
            "distr_loss": distr_loss,
            "accuracy": accuracy,
            "f1_score": f1_score
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Compute, log and return the desired losses and metrics during
        # validation.

        labels = batch["labels"].view(-1)
        outputs = self(batch["pixel_values"])
        logits = outputs["logits"].view(-1, self.num_classes)
        task_loss = self.loss_fn(logits, labels)
        rhos = outputs["rhos"]
        ponder_loss = self.ponder_loss_fn(rhos)
        halting_score_distr = outputs["halting_score_distr"]
        halting_score_distr_target = outputs["halting_score_distr_target"]
        distr_loss = self.distr_loss_fn(
            halting_score_distr, halting_score_distr_target
        )
        loss = (
            task_loss
            + self.alpha_ponder * ponder_loss
            + self.alpha_distr * distr_loss
        )
        accuracy = self.accuracy_fn(logits, labels)
        f1_score = self.f1_score_fn(logits, labels)

        self.log(
            "val_loss", loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "val_task_loss", task_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "val_ponder_loss", ponder_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "val_distr_loss", distr_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "val_accuracy", accuracy,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "val_f1_score", f1_score,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )

        return {
            "loss": loss,
            "task_loss": task_loss,
            "ponder_loss": ponder_loss,
            "distr_loss": distr_loss,
            "accuracy": accuracy,
            "f1_score": f1_score
        }

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        PyTorch Lightning hook.
        See PyTorch Lightning documentation for details.
        """

        # Compute, log and return the desired losses and metrics during
        # testing.
        
        labels = batch["labels"].view(-1)
        outputs = self(batch["pixel_values"])
        logits = outputs["logits"].view(-1, self.num_classes)
        task_loss = self.loss_fn(logits, labels)
        rhos = outputs["rhos"]
        ponder_loss = self.ponder_loss_fn(rhos)
        halting_score_distr = outputs["halting_score_distr"]
        halting_score_distr_target = outputs["halting_score_distr_target"]
        distr_loss = self.distr_loss_fn(
            halting_score_distr, halting_score_distr_target
        )
        loss = (
            task_loss
            + self.alpha_ponder * ponder_loss
            + self.alpha_distr * distr_loss
        )
        accuracy = self.accuracy_fn(logits, labels)
        f1_score = self.f1_score_fn(logits, labels)

        self.log(
            "test_loss", loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "test_task_loss", task_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "test_ponder_loss", ponder_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "test_distr_loss", distr_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "test_accuracy", accuracy,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "test_f1_score", f1_score,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )

        return {
            "loss": loss,
            "task_loss": task_loss,
            "ponder_loss": ponder_loss,
            "distr_loss": distr_loss,
            "accuracy": accuracy,
            "f1_score": f1_score
        }
