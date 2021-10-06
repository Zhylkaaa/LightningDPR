import pytorch_lightning as pl


class DPRDatasetModule(pl.LightningDataModule):
    def __init__(self):
        super(DPRDatasetModule, self).__init__()
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        pass

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        pass
