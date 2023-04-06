import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

class TrainPipeline:
    def __init__(self, config) -> None:
        self.gpt = config.model
        self.optimizer = optim.Adam(self.gpt.parameters(), lr=config.lr)
        self.loss = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.step_size)

    def step(self, batch_data):
        input, target = batch_data
        target = target.view(-1,)
        output = self.gpt(input).view(-1, config.vocab_size)
        self.optimizer.zero_grad()
        loss = self.loss(output.float(), target.squeeze())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def train(self, dataset, epoch=2):
        dataloader = DataLoader(dataset, batch_size=1)
        for _ in range(epoch):
            for batch in dataloader:
                self.step(batch)
        

if __name__ == "__main__":
    from dataset import MyDataset
    from data_pipeline import DataPipeline
    from config import GPTConfig

    old_data = './data/red_UTF82.txt'
    stop_words = './data/stop_words.txt'
    pipeline = DataPipeline(old_data, stop_words)
    contents, vocab_size = pipeline.process()

    dataset = MyDataset(contents, 512, 12)

    config = GPTConfig()

    pipeline = TrainPipeline(config)
    pipeline.train(dataset)
