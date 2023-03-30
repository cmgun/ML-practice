import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def get_num_leafs(my_tree):
    """
    calculate the leafs number
    :param my_tree:
    :return:
    """
    num_leafs = 0
    first_key = next(iter(my_tree))
    second_dict = my_tree[first_key]
    for key in second_dict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    """
    use recurrence to get tree depth
    :param my_tree:
    :return: the depth of the tree
    """
    max_depth = 0
    first_key = next(iter(my_tree))
    second_dict = my_tree[first_key]
    for key in second_dict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(second_dict[key]).__name__ == 'dict':
            current_depth = 1 + get_tree_depth(second_dict[key])
        else:
            current_depth = 1
        if current_depth > max_depth:
            max_depth = current_depth
    return max_depth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    plot tree node
    :param node_txt:
    :param center_pt:
    :param parent_pt:
    :param node_type:
    :return:
    """
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """
    plot the text on the arrow
    :param cntr_pt:
    :param parent_pt:
    :param txt_string:
    :return:
    """
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(my_tree, parent_pt, node_txt):
    """
    plot tree
    :param my_tree:
    :param parent_pt:
    :param node_txt:
    :return:
    """
    num_leafs = get_num_leafs(my_tree)  # this determines the x width of this tree
    depth = get_tree_depth(my_tree)
    first_str = next(iter(my_tree))  # the text label for this node should be this
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decisionNode)
    second_dict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leafNode)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def create_plot(in_tree, figsize=None):
    fig = plt.figure(1, figsize, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plot_tree.totalW = float(get_num_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()