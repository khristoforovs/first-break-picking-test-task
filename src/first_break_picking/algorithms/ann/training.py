import os
from first_break_picking.algorithms.ann import env
from first_break_picking.algorithms.ann.datasets.dataset import FirstBrakeDataset
from first_break_picking.algorithms.ann.models.model import loss, model
from first_break_picking.algorithms.ann.trainers.trainer import Trainer


dataset = FirstBrakeDataset.load(
    file_path=os.path.join(os.environ.get("DATASET_FOLDER")),
    validation_split=0.85,
)

trainer = Trainer(model=model, loss=loss)
trainer.train(dataset=dataset)


pass