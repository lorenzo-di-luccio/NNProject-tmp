import argparse
import pytorch_lightning
import pytorch_lightning.callbacks
import pytorch_lightning.loggers

import avit


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--batch_size", action="store", type=int, default=128
    )
    cli.add_argument(
        "--model", action="store", choices=["DeiT", "AViT"],
        required=True
    )
    cli.add_argument(
        "--lr", action="store", type=float, default=1.e-4
    )
    cli.add_argument(
        "--ckpt", action="store"
    )
    cli.add_argument(
        "--device", action="store", choices=["cpu", "gpu"], default="cpu",
    )
    args = cli.parse_args()

    print("Instantiating the Caltech-256 data module...")
    dm = avit.Caltech256DataModule(avit.HF_MODEL_NAME, args.batch_size)
    print("Caltech-256 data module instantiated.")

    print("Instantiating the chosen model...")
    if args.model == "DeiT":
        model = avit.DeiTModel(avit.AVIT_KWARGS, lr=args.lr)
    else:
        model = avit.AViTModel(avit.AVIT_KWARGS, lr=args.lr)
    print("Chosen model instantiated.")

    print("Loading DeiT weights from HuggingFace...")
    model.transf_model.load_hf_deit_weights(avit.HF_MODEL_NAME, avit.AVIT_KWARGS)
    print("DeiT weights from HuggingFace loaded.")

    print("Instantiating the trainer...")
    callbacks = [
        pytorch_lightning.callbacks.RichModelSummary(max_depth=-1),
        pytorch_lightning.callbacks.RichProgressBar(leave=False),
        pytorch_lightning.callbacks.EarlyStopping(
            monitor="val_loss_epoch", min_delta=0, patience=3, mode="min",
            strict=True, check_on_train_epoch_end=True
        ),
        pytorch_lightning.callbacks.ModelCheckpoint(
            dirpath="./model", filename="{epoch:03d}--{val_loss_epoch:.4f}",
            monitor="val_loss_epoch", mode="min",
            save_on_train_epoch_end=True
        )
    ]
    logger = pytorch_lightning.loggers.CSVLogger("./lightning", name="logs")
    trainer = pytorch_lightning.Trainer(
        accelerator=args.device,
        logger=logger,
        callbacks=callbacks,
        min_epochs=1,
        max_epochs=-1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accumulate_grad_batches=1,
        log_every_n_steps=25
    )
    print("Trainer instantiated.")

    pytorch_lightning.seed_everything(seed=avit.SEED, workers=True)

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt)
