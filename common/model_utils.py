import matplotlib.pyplot as plt

def plot_accuracy(history):
    history_dict = history.history
    accuracies = history_dict['acc']
    validation_accuracies = history_dict['val_acc']
    epochs = range(1, len(accuracies) + 1)
    plt.plot(epochs, accuracies, 'bo')
    plt.plot(epochs, validation_accuracies, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Train and Validation accuracy')
    plt.show()

def plot_loss(history):
    history_dict = history.history
    losses = history_dict['loss']
    validation_losses = history_dict['val_loss']
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, 'bo')
    plt.plot(epochs, validation_losses, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Train and Validation loss')
    plt.show()
