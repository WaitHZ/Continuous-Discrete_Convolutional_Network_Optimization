import matplotlib.pyplot as plt


class DynamicDrawer(object):
    """
        A dynamic image drawing class based on matplotlib.
    """
    def __init__(self, num_plots, lengends, x_lim, y_lim) -> None:
        self.x_list = []
        self.y_s_list = [[] for i in range(num_plots)]
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.legends = lengends

    def add_points(self, x, y_s):
        plt.clf()

        self.x_list.append(x)
        plt.xlim(self.x_lim)
        plt.ylim(self.y_lim)

        for i, y in enumerate(y_s):
            self.y_s_list[i].append(y)
            plt.plot(self.x_list, self.y_s_list[i], label=self.legends[i])

        plt.grid(alpha=0.5)
        plt.legend()

    def save_fig(self, root='./fig/', name='fig'):
        plt.savefig(root+name+'.png')
