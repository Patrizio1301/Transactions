import numpy as np
import matplotlib.pyplot as pltp
import matplotlib.dates as mdates
from datetime import datetime
from time import mktime


class Timeline(object):

    def __init__(self, df):
        self.dates = df['date']
        self.names = df['transaction']

    def __call__(self, *args, **kwargs):
        levels = np.tile([-5, 5, -3, 3, -1, 1],
                         int(np.ceil(len(self.dates)/6)))[:len(self.dates)]

        # Create figure and plot a stem plot with the date
        fig, ax = pltp.subplots(figsize=(8.8, 4))
        ax.set(title="Transaction timeline")

        dates = [mktime(datetime.strptime(i, "%Y-%m-%d").timetuple()) for i in self.dates]
        markerline, stemline, baseline = ax.stem(dates, levels,
                                                 linefmt="C3-", basefmt="k-",
                                                 use_line_collection=True)

        pltp.setp(markerline, mec="k", mfc="w", zorder=3)

        # Shift the markers to the baseline by replacing the y-data by zeros.
        markerline.set_ydata(np.zeros(len(self.dates)))

        # annotate lines
        vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
        for d, l, r, va in zip(dates, levels, self.names, vert):
            ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),
                        textcoords="offset points", va=va, ha="right")

        # format xaxis with 4 month intervals
        ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=2))
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
        pltp.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # remove y axis and spines
        ax.get_yaxis().set_visible(False)
        for spine in ["left", "top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.margins(y=0.1)

        pltp.savefig('timeline.png')
        pltp.gcf().clear()

        return pltp

"""
Function used to visualize the training history
metrics: Visualized metrics,
save: if the png are saved to disk
history: training history to be visualized
"""
def plot_training_info(metrics, save, history):
    # summarize history for accuracy
    if 'acc' in metrics:
        pltp.plot(history['acc'])
        pltp.plot(history['val_acc'])
        pltp.title('model accuracy')
        pltp.ylabel('accuracy')
        pltp.xlabel('epoch')
        pltp.legend(['train', 'test'], loc='upper left')
        if save == True:
            pltp.savefig('accuracy.eps')
            pltp.gcf().clear()
        else:
            pltp.show()

    # summarize history for loss
    if 'loss' in metrics:
        pltp.plot(history['loss'])
        pltp.plot(history['val_loss'])
        pltp.title('model loss')
        pltp.ylabel('loss')
        pltp.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        pltp.yscale("log")
        pltp.legend(['train', 'test'], loc='upper left')
        if save == True:
            pltp.savefig('loss.eps')
            pltp.gcf().clear()
        else:
            pltp.show()

    if 'output_weekday_acc' in metrics:
            pltp.plot(history['output_weekday_acc'])
            pltp.plot(history['val_output_weekday_acc'])
            pltp.title('model accuracy')
            pltp.ylabel('accuracy')
            pltp.xlabel('epoch')
            pltp.legend(['train', 'test'], loc='upper left')
            if save == True:
                pltp.savefig('output_weekday_acc.eps')
                pltp.gcf().clear()
            else:
                pltp.show()

    if 'output_transaction_acc' in metrics:
        pltp.plot(history['output_transaction_acc'])
        pltp.plot(history['val_output_transaction_acc'])
        pltp.title('model accuracy')
        pltp.ylabel('accuracy')
        pltp.xlabel('epoch')
        pltp.legend(['train', 'test'], loc='upper left')
        if save == True:
            pltp.savefig('output_transaction_acc.eps')
            pltp.gcf().clear()
        else:
            pltp.show()
