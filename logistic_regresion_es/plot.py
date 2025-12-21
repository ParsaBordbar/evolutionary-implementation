from matplotlib import pyplot as plt


def plot_training_curves(best_losses, mean_losses, train_acc):
    plt.figure()
    plt.plot(best_losses, label='Best Loss')
    plt.plot(mean_losses, label='Mean Loss')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_acc)
    plt.xlabel('Generation')
    plt.ylabel('Training Accuracy')
    plt.show()