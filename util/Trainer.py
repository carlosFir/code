import os
from .logger import get_logger
import torch
import time
import copy


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader, test_loader,
                 args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.log_dir = args['train']['log_dir']
        self.log_step = int(args['train']['log_step'])
        self.lr_decay = bool(args['train']['lr_decay'])
        self.epochs = int(args['train']['epochs'])
        self.early_stop = bool(args['train']['early_stop'])
        self.early_stop_patience = int(args['train']['early_stop_patience'])
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.log_dir, 'best_model_test.pth')
        self.loss_figure_path = os.path.join(self.log_dir, 'loss.png')
        #log
        if os.path.isdir(self.log_dir) == False:
            os.makedirs(self.log_dir, exist_ok=True)
        self.logger = get_logger(self.log_dir)
        self.logger.info('Experiment log path in: {}'.format(self.log_dir))
    

    def val_epoch(self, model, epoch, val_dataloader):
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (props, seqs, attention_mask) in enumerate(val_dataloader):
                props = props.to(self.model.device)
                seqs = seqs.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)
                labels = [seqs, props]
                data = [props, seqs[..., :-1]]
                output = model(data, labels, attention_mask)
                loss = output.loss
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        #b = time.time()
       
        for batch_idx, (props, seqs, attention_mask) in enumerate(self.train_loader):
            props = props.to(self.model.device)
            seqs = seqs.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
            labels = [seqs, props]
            data = [props, seqs[..., :-1]] # 丢弃end token
            self.optimizer.zero_grad()
           
            output = self.model(data, labels, attention_mask)

            loss = output.loss
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
       
        train_epoch_loss = total_loss/self.train_per_epoch
        #print('epoch time cost : {}'.format(time.time()-b))
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, 0))

        #learning rate decay
        # if self.lr_decay:
        #     self.lr_scheduler.step()
        return train_epoch_loss
    
    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(self.model, epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.early_stop:
                if not_improved_count == self.early_stop_patience:
                    best_model = self.finetune(best_loss, best_model, epoch)
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(self.model, self.best_path)
                self.logger.info("Saving current best --- whole --- model to " + self.best_path)

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        #test
        self.model.load_state_dict(best_model)
        self.val_epoch(self.model, self.epochs, self.test_loader)