from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RandomData(Dataset):
    def __init__(self):
        self.dataset = np.random.rand(1000,64).astype(np.float32)
        self.target = np.random.randint(0,2,1000).astype(np.float32)
    
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        item = {
            "data": self.dataset[idx],
            "target": self.target[idx]
        }
        return item

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # NN Architecture
        self.fc = nn.Linear(64,1)
    
    def forward(self, x):
        """
        Method used for inference
        """
        pred = self.fc(x)
        return pred
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        """
        Training step
        """
        x = train_batch["data"]
        y = train_batch["target"]      
        out = self.fc(x)
        out = F.sigmoid(out)
        loss = F.binary_cross_entropy(out, y.view(-1, 1))
        self.logger.experiment.log_metric({"train_loss": loss})
        return loss
    
dataset = RandomData()
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
model = SimpleModel()
mlf_logger = MLFlowLogger(experiment_name="a_naive_experiment2", tracking_uri="file:./ml-runs")
trainer = Trainer(logger=mlf_logger, max_epochs=1)
trainer.fit(model, train_loader)
