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
        "--ckpt", action="store", required=True
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

    print("Instantiating the trainer...")
    callbacks = [
        pytorch_lightning.callbacks.RichModelSummary(max_depth=-1),
        pytorch_lightning.callbacks.RichProgressBar(leave=False)
    ]
    trainer = pytorch_lightning.Trainer(
        accelerator=args.device,
        logger=None,
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

    trainer.test(model, datamodule=dm, ckpt_path=args.ckpt)
