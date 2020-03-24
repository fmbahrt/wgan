import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
