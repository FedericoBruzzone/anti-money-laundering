import matplotlib.pyplot as plt

def print_hello():
    print('Hello')

def print_goodbye():
    print('Goodbye')

def plot_first_graph():
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    
    if plt.get_backend().lower() == 'module://ipykernel.pylab.backend_inline':
        plt.show()
    else:
        plt.savefig("plt_figs/mygraph.png")
        plt.close()

plot_first_graph()
