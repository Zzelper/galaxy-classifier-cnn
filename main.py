from src.data_import import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #print("Hello World")
    images, labels = load_images_labels()
    #train_images, train_labels, test_images, test_labels = split_dataset(images, labels)
    print(images[0])

    test_image = images[0].detach().cpu().numpy()/256
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(test_image)
    plt.show()
