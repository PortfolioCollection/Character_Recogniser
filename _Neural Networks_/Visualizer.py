#CREDIT TO https://gist.github.com/craffel/2d727968c3aaebd10359

import math
import matplotlib.pyplot as plt
from mpldatacursor import datacursor


def draw_neural_net(left, right, bottom, top, layer_sizes, layer_text=None, weights = None):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.set_autoscaley_on(False)
    ax.set_autoscalex_on(False)
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes))
    #ax.axis('off')
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            x = n*h_spacing + left
            y = layer_top - m*v_spacing
            circle = plt.Circle((x,y), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            # Node annotations
            if layer_text:
                text = layer_text.pop(0)
                text_size = 700*v_spacing/(4.*(math.log(len(str(text))+1,7)))
                plt.annotate(text, xy=(x, y), zorder=5, ha='center', va='center',size=text_size)
                #plt.annotate("Val:"+str(text), xy=(x, y), zorder=5, ha='center', va='center',size=text_size)
                #plt.annotate("Err:"+str(text), xy=(x, y-0.1), zorder=5, ha='center', va='center',size=text_size)

    # Edges
    count = 0
    weight = 0
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                #line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                #                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                #ax.add_artist(line)
                if weights:
                    if count < len(weights):
                        weight = weights[count]
                        count+=1
                ax.plot([n*h_spacing + left, (n + 1)*h_spacing + left],
                        [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing],
                        'k',
                        label='$Weight = {}$'.format(weight))

    datacursor(display='multiple', draggable=True , formatter='{label}'.format, bbox=dict(fc='yellow', alpha=1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Need empty strings for unlabeled nodes at start, but not at end
    node_text = ['I','s','I','t','hanifheihiehfihefowefhowehfowhefofdjaojef[aejgf[aejfajefoja[fej[aeofjpi']
    weights = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    draw_neural_net(0.1, 1, 0, 1, [2,2,4,3,1], node_text, weights)
