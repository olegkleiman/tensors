import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

initial_weight = 6
initial_bias = -4

current_weight = initial_weight
current_bias = initial_bias

sigmoid = lambda x, w, b: 1. / (1. + np.exp(-(x * w + b)))
x_input = np.linspace(-2., 2, 100)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'

plt.axes(ax)  # select sin_ax
plt.title(r'$y=\frac{1}{1 + e^{-(wx+b)}}$') # 'y = sigmoid(x, weight, bias)')
sigmoid_plot, = plt.plot(x_input, sigmoid(x_input, initial_weight, initial_bias))

slider_bias_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
slider_weight_ax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

bias_slider = Slider(slider_bias_ax, 'bias', -10., 10., valinit=initial_bias)
weight_slider = Slider(slider_weight_ax, 'weight', -100., 100., valinit=initial_weight)


def change_plot():
    sigmoid_plot.set_ydata(sigmoid(x_input, current_weight, current_bias))
    fig.canvas.draw_idle()


def update(new_bias):
    global current_bias
    current_bias = new_bias
    change_plot()


def weight_update(new_weight):
    global current_weight
    current_weight = new_weight
    change_plot()


bias_slider.on_changed(update)
weight_slider.on_changed(weight_update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    bias_slider.reset()
    weight_slider.reset()


button.on_clicked(reset)

plt.show()
