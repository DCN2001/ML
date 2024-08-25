from config import get_config
import dataloader.moisesdb_loader as dataloader
from model.Trans_Attrc import Trans_Attrc 
print("hello")
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp
from torchinfo import summary
import torch.nn as nn  
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from tqdm import tqdm

args = get_config()
class Trainer():
    def __init__(self, train_loader, test_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.valid_loader = test_loader
        self.model = Trans_Attrc(kernel_size=16, channels=256, chunk_size=250,n_heads=8, n_intra=1,n_inter=1,r=4).to(self.device)   #n_intra=4, n_inter=2 out of memory
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        #summary(self.model, [(1, 2, 132300),1])   
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr, weight_decay=args.l2_lambda)

    @torch.no_grad()
    def valid_batch(self, batch):
        model_input = batch[0].to(self.device)
        num_instru = batch[1]
        loss = self.model(model_input,num_instru)
        loss = loss.mean()
        return loss.item()
    
    def valid_total(self):
        loss_total = 0.0
        for idx, batch in enumerate(tqdm(self.valid_loader, desc="Eval bar", colour="#9F35FF")):
            step = idx + 1
            loss_batch = self.valid_batch(batch)
            loss_total += loss_batch
        loss_total = loss_total/step
        return loss_total

    def train_batch(self, batch):
        #assign to device
        model_input = batch[0].to(self.device)
        num_instru = batch[1]   
        #start
        loss = self.model(model_input, num_instru)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 5)  #Prevent gradient explosion
        self.optimizer.step()
        return loss.item()
        
    def train_total(self):
        train_loss_list = []
        valid_loss_list = []
        #scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_epoch, gamma=0.75)

        for epoch in tqdm(range(args.epochs), desc="Epoch", colour="#0080FF"):
            self.model.train()
            train_loss = 0.0
            for idx, batch in enumerate(tqdm(self.train_loader, desc=f"Train bar({epoch})", colour="#ff7300")):
                step = idx + 1
                loss_batch = self.train_batch(batch)
                train_loss += loss_batch
            train_loss_list.append(train_loss)
            print(f"\n train loss: {train_loss/step}")

            self.model.eval()
            valid_loss = self.valid_total()
            valid_loss_list.append(valid_loss)
            print(f"\n valid loss: {valid_loss}")
            #Validation (every 5 epoch)
            if epoch%5==0:
                saved_model_path = args.model_save_path + "_" + str(epoch) +".pth"
                print("Saving model............")
                torch.save(self.model.state_dict(), saved_model_path)
            #scheduler.step()
        
        #Draw figure
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss', color='blue')
        plt.plot(range(len(valid_loss_list)), valid_loss_list, label='valid_loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss vs. Epoch best')
        plt.legend()
        plt.grid(True)
        plt.savefig('./curve_plot/loss.png')  # 保存准确率图
        plt.close()
        
#------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    datapath = args.data_path
    print("Loading npy file.......")
    data = np.load(datapath, allow_pickle=True).item()
    train_loader, valid_loader = dataloader.load_data(data, args.batch_size, 1)   #number of gpu
    trainer = Trainer(train_loader, valid_loader)
    trainer.train_total()


'''
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank:int, world_size:int, args):
    setup(rank,world_size)
    datapath = args.data_path
    print("Loading npy file.......")
    data = np.load(datapath, allow_pickle=True).item()
        
    train_loader, valid_loader = dataloader.load_data(data, args.batch_size, world_size)   #number of gpu
    trainer = Trainer(train_loader, valid_loader, rank, world_size)
    trainer.train_total()
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
'''


    
    




