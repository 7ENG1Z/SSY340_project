import torch
import matplotlib.pyplot as plt


class TrainModel():
    def __init__(self,model,device,criterion,optimizer):
        self.model=model
        self.criterion=criterion
        self.optimizer=optimizer
        self.device=device
        

    def fit(self,num_epochs,train_loader,val_loader):
        print("Start training!")
        
        self.model.to(self.device)
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        for epoch in range(1,num_epochs+1):
            train_loss,train_acc = self.training_epoch(train_loader)
            val_loss,val_acc =self.validate(val_loader)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}/{num_epochs}:"
                      f"Train loss: {train_loss:.3f},"
                      f"Train acc: {train_acc:.3f},"
                      f"Val loss: {val_loss:.3f},"
                      f"Val acc.: {val_acc:.3f}"
                     )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        # self.model.to('cpu') if we want model back to cpu

        return self.model, train_losses, train_accs, val_losses, val_accs

    def training_epoch(self,train_loader):
        self.model.train()

        train_loss_batches, train_acc_batches = [], []
        for i, (inputs, labels) in enumerate(train_loader):
            # toGPU
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            #forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)   
            #backward and step
            loss.backward()
            self.optimizer.step()

            #change the output to index lable and caculate the acc
            pred_label = torch.argmax(outputs, dim=1)
            acc= (pred_label == labels).float().mean().item()

            train_loss_batches.append(loss.item())
            train_acc_batches.append(acc)

        return sum(train_loss_batches)/len(train_loss_batches),sum(train_acc_batches)/len(train_acc_batches)
    
    def validate(self,val_loader):
        val_loss_sum = 0
        val_acc_sum = 0
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
            # toGPU
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)   
                val_loss_sum += loss.item()

                pred_label = torch.argmax(outputs, dim=1)
                acc= (pred_label == labels).float().mean().item()
                
                val_acc_sum += acc
        return val_loss_sum/len(val_loader), val_acc_sum/len(val_loader)
    
    


def validation(model,val_loader):
        val_acc_sum = 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
            # toGPU
                outputs = model(inputs)
                pred_label = torch.argmax(outputs, dim=1)
                acc= (pred_label == labels).float().mean().item()
                val_acc_sum += acc
                
        return val_acc_sum/len(val_loader)

def show_result(train_losses,train_accs,val_losses,val_accs):
        # YOUR CODE HERE
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    #Plot losses in training and validation
    axes[0].plot((range(1, len(train_losses)+1)), train_losses, label='Training Loss', color='blue', marker = 'o')
    axes[0].plot((range(1, len(val_losses)+1)), val_losses, label='Validation Loss', color='red', marker = 'x')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training and Validation Loss of LSTM')

    #Plot accuracy
    axes[1].plot((range(1, len(train_accs)+1)), train_accs, label='Training accuracy', color='blue', marker = 'o')
    axes[1].plot((range(1, len(val_accs)+1)), val_accs, label='Validation accuracy', color='red', marker = 'x')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Training and Validation Accuracy of LSTM')

    plt.tight_layout()
    plt.show()

    def freeze(self, layer_name):
        # 冻结指定层
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

    def unfreeze(self, layer_name):
        # 解冻指定层
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = True





