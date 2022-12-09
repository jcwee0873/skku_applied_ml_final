from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

    
class Trainer(nn.Module):
    def __init__(self, model, crit, optimizer):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.crit = crit.to(self.device)
        self.optimizer = optimizer
        
        self.train_history = []
        self.valid_history = []
        
    def _train(self, X, y, epoch, batch_size=512):
        train_loss = 0
        
        indices = torch.randperm(X.size(0))
        X = torch.index_select(X, 0, indices)
        y = torch.index_select(y, 0, indices)
        
        X = X.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)
        y_hat = []
        h = []
                
        for X_i, y_i in zip(X, y):
            X_i = X_i.to(self.device)
            y_i = y_i.to(self.device)
            
            y_hat_i, h_i = self.model(X_i)
            loss = self.crit(y_hat_i, y_i)
            loss = loss.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            train_loss += float(loss)

            h += [h_i.detach().cpu().numpy()]
            y_hat += [y_hat_i.detach().cpu().numpy()]  

        h = np.concatenate(h, axis=0)
        y_hat = np.concatenate(y_hat, axis=0)
        y = np.concatenate([i.detach().cpu().numpy() for i in y], axis=0)
        self.plot_tsne_plot(epoch, h, y, y_hat)

        return train_loss / len(X)
    
    def _valid(self, X, y, batch_size=512):
        valid_loss = 0
        X = X.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)
        
        with torch.no_grad():
            for X_i, y_i in zip(X, y):
                X_i = X_i.to(self.device)
                y_i = y_i.to(self.device)

                y_hat_i, z = self.model(X_i)
                loss = self.crit(y_hat_i, y_i)
                loss = loss.mean()

                valid_loss += float(loss)
            
        return valid_loss / len(X)
    
    
    def plot_loss_history(self, figsize=(20, 10)):
        plt.figure(figsize=figsize)
        plt.grid(True)
        plt.title('Train / Valid Loss History')
        plt.plot(range(0, len(self.train_history)), self.train_history, label='Train')
        plt.plot(range(0, len(self.valid_history)), self.valid_history, label='Valid')
        plt.yscale('log')
        plt.legend()
        plt.show()
    

    def plot_tsne_plot(
        self,
        epoch,
        h, y, y_hat,
        classes=['Angry', 'Happy', 'Sad', 'Neutral']
    ):
        if not self.ax is None:
            tsne = TSNE(
                n_components=2, perplexity=30.0, early_exaggeration=12.0,
                learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
                min_grad_norm=1e-07, metric='euclidean', init='pca', verbose=0,
                random_state=None, method='barnes_hut', angle=0.5
            )

            df = pd.DataFrame(tsne.fit_transform(h), columns=['x', 'y'])
            df['Real'] = [classes[i] for i in y]
            df['Pred'] = [classes[i] for i in y_hat.argmax(axis=-1)]
            sns.scatterplot(df.sort_values('Real'), x='x', y='y', hue='Real', alpha=.3, ax=self.ax[epoch-1, 0])
            sns.scatterplot(df.sort_values('Pred'), x='x', y='y', hue='Pred', alpha=.3, ax=self.ax[epoch-1, 1])


    def train(
        self, 
        train_data,
        valid_data, 
        epochs=10, 
        batch_size=512,
        print_interval=1, 
        early_stop=5,
        figsize=(20, 10)
    ):        
        best_model = None
        lowest_loss = np.inf
        lowest_epoch = 0

        if figsize:
            self.fig, self.ax = plt.subplots(nrows=epochs, ncols=2, figsize=figsize)
        else:
            self.ax = None
            
        for epoch in range(1, epochs + 1):
            train_loss = self._train(train_data[0], train_data[1], epoch, batch_size=batch_size)
            valid_loss = self._valid(valid_data[0], valid_data[1], batch_size=batch_size)

            self.train_history += [train_loss]
            self.valid_history += [valid_loss]
        
            if valid_loss <= lowest_loss:
                best_model = deepcopy(self.model.state_dict())
                lowest_loss = valid_loss
                lowest_epoch = epoch
                
            else:
                if (epoch - lowest_epoch) >= early_stop:
                    print("Early Stop at Epoch {epoch}. Lowest Loss {lowest_loss} at {lowest_epoch} epoch".format(
                        epoch=epoch,
                        lowest_loss=round(lowest_loss, 5),
                        lowest_epoch=lowest_epoch
                    ))
                    break
                
            if epoch % print_interval == 0:
                print("Epoch {epoch}: train_loss={train_loss}, valid_loss={valid_loss}, lowest_loss={lowest_loss}".format(
                    epoch=epoch,
                    train_loss=round(train_loss, 5),
                    valid_loss=round(valid_loss, 5),
                    lowest_loss=round(lowest_loss, 5)
                ))
                
        print('The best validstion loss from epoch={epoch}, loss={lowest_loss}'.format(
            epoch=lowest_epoch,
            lowest_loss=round(lowest_loss, 5)
        ))
        
        self.model.load_state_dict(best_model)