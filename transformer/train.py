import argparse
import lightning as L
from transformer.model import WMT14TransformerModule
from transformer.dataset import WMT14DataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

def get_arguments():
    parser = argparse.ArgumentParser(description="Train Transformer from English to Geman")

    # Data
    parser.add_argument("--data_dir", type=str, default="./data")

    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)

    # Logging and Checkpoint
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="transformer_base")

    # Hardware
    parser.add_argument("--accelerator", type=str, default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    return parser.parse_args()


def main():
    args = get_arguments()

    print("="*50)
    print("Training Configuration")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Dataset
    datamodule = WMT14DataModule(
        train_src_file=f"{args.data_dir}/processed/train.src",
        train_tgt_file=f"{args.data_dir}/processed/train.tgt",
        val_src_file=f"{args.data_dir}/processed/valid.src",
        val_tgt_file=f"{args.data_dir}/processed/valid.tgt",
        src_tokenizer_file=f"{args.data_dir}/tokenizers/en_tokenizer.model",
        tgt_tokenizer_file=f"{args.data_dir}/tokenizers/de_tokenizer.model",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len
    )
    datamodule.setup()

    # Model
    model = WMT14TransformerModule(
            src_vocab_size=len(datamodule.src_tokenizer),
            tgt_vocab_size=len(datamodule.tgt_tokenizer),
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_len=args.max_len,
            warmup_steps=args.warmup_steps,
            label_smoothing=args.label_smoothing,
            pad_idx=datamodule.tgt_tokenizer.pad_idx,

    )

    print("="*50)
    print("Model Parameters")
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        num = p.numel()
        total_params += num
        if p.requires_grad:
            trainable_params += num
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.checkpoint_dir}/{args.experiment_name}",
        filename="transformer-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        devices="auto",
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.5, # Validate twice per epoch
        deterministic=False, 
    )
    trainer.fit(model, datamodule)


    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()