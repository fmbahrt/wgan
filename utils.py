import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import imageio
import numpy as np
sns.set(style="darkgrid")

class Visualizer:

    def __init__(self, save_path):
        self.path = save_path
        self.df   = pd.DataFrame()

    def append_data(self, data):
        dft = pd.DataFrame(data)
        self.df = self.df.append(dft, ignore_index=True)

    def plot_and_save(self, x_name, y_name):
        sns_plot = sns.relplot(x=x_name, y=y_name, ci=None, kind="line",
                               data=self.df)
        sns_plot.savefig(self.path)

        # index
        path = os.path.split(self.path)
        name = path[1].split('.')[0]
        csv_path = os.path.join(path[0], name+".csv")
        self.df.to_csv(csv_path, index=False, header=True)

class GANVisualizer:

    def __init__(self, dirz, run_name):
        check_dir = os.path.join(dirz, run_name)
        if not os.path.exists(check_dir):
            os.mkdir(check_dir)

        self.check_dir = check_dir

        self.losses = []
        self.imgs   = []
        self.i = []

        self.gif_name = 'seed_history.gif'
        self.loss_file = 'loss_history.npz'

    def add_step(self, i, loss, img):
        self.losses.append(loss)
        self.i.append(i)
        img = (img.permute(1, 2, 0).data.cpu().numpy() * 255).astype(np.uint8)
        self.imgs.append(img)

    def commit(self):

        # plot and save
        plt.plot(self.i, self.losses)
        plt.xlabel("Generator Iterations")
        plt.ylabel("Critic Loss")
        plt.savefig(os.path.join(self.check_dir, "loss_history.png"))

        np.save(os.path.join(self.check_dir, self.loss_file),
                np.array(self.losses))
        imageio.mimsave(os.path.join(self.check_dir, self.gif_name), self.imgs)
