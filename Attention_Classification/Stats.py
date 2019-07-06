
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Attention_Classification.utils import save_model, save_stats


class Stats:

    def __init__(self, config, patience=15):
        self.loss_history = []

        self.wait = 0
        self.stop_training = False
        self.best_loss = 50
        self.patience = patience

        self.config = config

        self.last_epoch = 0

    def on_epoch_end(self, model, epoch, train_loss, dev_loss):
        self.last_epoch = epoch

        self.loss_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'dev_loss': dev_loss
        })

        if dev_loss - self.best_loss < 0:
            
            self.best_loss = dev_loss
            self.wait = 1
            self.save_on_epoch_end(model)
        else:
            if self.wait >=self.patience:
                self.stop_training = True
                self.stopped_epoch = epoch
            self.wait += 1
        self.plot_loss()

    def on_train_end(self):
        print('Training Terminated for Early Stopping at Epoch {}'.format(self.stopped_epoch))

    def save_on_epoch_end(self, model):
        print('Saving Model !!!!')
        save_model(model, self.config['n_gpu'], self.config['model_dir'], 'best_model.pth')

    def plot_loss(self):
        df = pd.DataFrame(self.loss_history)
        df.sort_values(by='epoch', inplace=True)

        plot = sns.lineplot(x='epoch', y='train_loss', data=df)
        fig = plot.get_figure()
        # fig.savefig("train_loss.png")

        plot = sns.lineplot(x='epoch', y='dev_loss', data=df)
        fig = plot.get_figure()
        plt.legend(['Train_Loss', 'Val_Loss'], loc='upper left')
        fig.savefig("Loss_plot.png")
