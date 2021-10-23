from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def get_default_callbacks(args):
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='dpr_{epoch}-{step}',
        monitor=args.monitor_metric,
        verbose=True,
        save_top_k=-1,
        mode='max',
        every_n_epochs=1,
    )
    if checkpoint_callback:
        callbacks.append(checkpoint_callback)

    callbacks.append(LearningRateMonitor(logging_interval='step'))

    return callbacks