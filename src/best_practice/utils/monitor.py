import contextlib
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import warnings
import os
from pathlib import Path
import platform

__all__ = ('PlotMonitor', 'ConfusionMatrix')

def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


class TryExcept(contextlib.ContextDecorator):
    # YOLOv5 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=''):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True
class _Monitor(object):

    def __init__(self, path) -> None:
        self.time_str: str = time.strftime("%Y-%m-%d-%H:%M:%S",
                                           time.localtime())
        self.path: str = path
        if not os.path.exists(self.path):
            os.mkdir(self.path)


class _Line():

    def __init__(self, color) -> None:
        self.x_len: int = 0
        self.y: np.ndarray = np.zeros(shape=0, dtype=np.float32)
        self.color = color


class PlotMonitor(_Monitor):

    def __init__(self, save_path="./") -> None:
        super(PlotMonitor, self).__init__(save_path)
        self.lines: dict[str, _Line] = {}

    def add_line(self, line_label: str, color: str = None):
        c = color if color != None else (
            random.uniform(0.4, 1), random.uniform(0.4, 1),
            random.uniform(0.4, 1))  # Give a random RGB color
        self.lines[line_label] = _Line(color=c)

    def check_line_existence(self, line_label: str):
        return True if line_label in self.lines.keys() else False

    def add_data(self, line_label: str, new_value: float) -> None:
        try:
            self.lines[line_label].y = np.append(self.lines[line_label].y,
                                                 new_value)
            self.lines[line_label].x_len += 1
        except KeyError:
            warnings.warn(f"line \"{line_label}\" is not added")

    def generate_graph(self,
                       pic_name="1.jpg",
                       title: str = None,
                       xlabel: str = None,
                       ylabel: str = None) -> None:
        if len(self.lines) == 0:
            warnings.warn("No line has been added")
            return

        fig, axes = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
        for l in self.lines:
            x = np.linspace(0, 1, self.lines[l].x_len)
            axes.plot(x,
                      self.lines[l].y,
                      linestyle="solid",
                      color=self.lines[l].color,
                      marker='.',
                      label=l)
            axes.set_xlabel(xlabel)
            axes.set_ylabel(ylabel)
            axes.grid(axis='y', color='0.95')
        fig.legend(loc="upper left", title='Parameter where:')
        fig.suptitle(title)

        # save figure to file
        if os.path.splitext(pic_name)[1] not in (".jpg", ".png"):
            raise ValueError(
                f"Only support jpg or png picture, but receive {pic_name}")
        fig.savefig(os.path.join(self.path, pic_name),
                    facecolor='grey',
                    edgecolor='red',
                    dpi=250)
        plt.close(fig)

    def reset(self):
        self.x = np.zeros(shape=0, dtype=np.float32)
        self.y = np.zeros(shape=0, dtype=np.float32)
        self.cnt = 0

class ConfusionMatrix(object):

    def __init__(self, nc, conf=0.25):
        self.matrix: np.ndarray = np.zeros((nc, nc))
        self.nc: int = nc  # number of classes
        self.conf: float = conf

    def process_batch(self, prediction: np.ndarray, labels: np.ndarray):
        # print(prediction, labels)
        for i, l in enumerate(labels):
            self.matrix[prediction[i], l] += 1  # correct

    def get_matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def plot(self, normalize=True, save_dir='', names=()):
        import seaborn as sn

        array = self.matrix / (
            (self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1
        )  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore'
            )  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           "size": 8
                       },
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_ylabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.nc):
            print('\t'.join(map(str, self.matrix[i])))



def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(
                px, y, linewidth=1,
                label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px,
            py.mean(1),
            linewidth=3,
            color='blue',
            label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    import numpy as np

    p: PlotMonitor = PlotMonitor()

    p.add_line("acc")
    p.add_line("time", color="red")

    for i in range(50):
        p.add_data("acc", i)

    for i in range(120):
        p.add_data("time", np.sin(i / np.pi))

    p.generate_graph("asdf.jpg",
                     title="this is a title",
                     xlabel="time",
                     ylabel="hahha")
