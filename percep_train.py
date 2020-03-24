from dataset import twoafc
from perceptual_cost import CostTrainingHarness

model = CostTrainingHarness(cuda=False)
model.train(twoafc(['./dataset/2afc/train/cnn'
                    './dataset/2afc/train/traditional',
                    './dataset/2afc/train/mix'], batch_size=128), epochs=20)
