import toml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
import sys

from experiments.circular_padding.resnet_circular_padding import *
from src import *
from src.data.data_module import MusicDataModule
from torch.utils.data import Dataset

config = toml.load('hindustani.toml')
fcd_h = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])))

config = toml.load('carnatic.toml')
fcd_c = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])),
                        carnatic=True)

train_h, fcd_not_train_h = fcd_h.greedy_split(train_size=0.70)
val_h, test_h = fcd_not_train_h.greedy_split(test_size=0.5)

train_c, fcd_not_train_c = fcd_c.greedy_split(train_size=0.70)
val_c, test_c = fcd_not_train_c.greedy_split(test_size=0.5)


class CombinedDataSet(Dataset):

    def __init__(self,ds1,ds2):
        self.ds1=ds1
        self.ds2=ds2
        self.l1=len(self.ds1)
        self.l2=len(self.ds2)
        self.raga_ids = dict(self.ds1.raga_ids)
        for k,v in self.ds2.raga_ids.items():
            self.raga_ids[k]=v+self.l1

    def __len__(self):
        return self.l1 + self.l2

    def __getitem__(self, item):
        if item<self.l1:
            return self.ds1[item]
        else:
            x,y=self.ds2[item-self.l1]
            # TODO: hardcoded to Hindustani number of ragas
            return x,30+y

train = CombinedDataSet(train_h,train_c)
val = CombinedDataSet(val_h,val_c)
test = CombinedDataSet(test_h,test_c)


train = ChromaChunkDataset(train, chunk_size=100, augmentation=None, stride=10)
data = MusicDataModule(train, val, batch_size=32)


num_classes = max(fcd_h.y)+max(fcd_c.y)+2
if sys.argv[1] == '34':
    model = ResNet34Circular(num_classes=num_classes)
elif sys.argv[1] == '50':
    model = ResNet50Circular(num_classes=num_classes)
elif sys.argv[1] == '101':
    model = ResNet101Circular(num_classes=num_classes)

model.epochs = 50

logger = WandbLogger(project='Raga Benchmark', name=f'Combined Training - ResNet{sys.argv[1]}')

checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    filepath=f'/mnt/disks/checkpoints/new-checkpoints/combined-resnet{sys.argv[1]}-{{epoch:02d}}-{{val_accuracy:.2f}}',
    save_top_k=1,
    mode='max',
    verbose=True
)

if len(sys.argv) == 3:
    print('Resuming!')
    resume_from_checkpoint = os.path.join('/mnt/disks/checkpoints/checkpoints/new-checkpoints', sys.argv[2])
else:
    print('Not resuming!')
    resume_from_checkpoint = None

trainer = Trainer(gpus=1, logger=logger, max_epochs=model.epochs, num_sanity_val_steps=2,
                  deterministic=True, resume_from_checkpoint=resume_from_checkpoint,
                  val_check_interval=0.1, auto_lr_find=False,checkpoint_callback=checkpoint_callback)

model.lr = 0.1
trainer.fit(model, data)