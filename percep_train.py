from dataset import twoafc
from perceptual_cost import CostTrainingHarness

model = CostTrainingHarness(cuda=False)
model.train(twoafc('./dataset/2afc/train/traditional', batch_size=32), epochs=1)
