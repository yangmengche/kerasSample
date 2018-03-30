import matplotlib.pyplot as plt

def plot(hisDict):
  plt.figure()
  pltIndex=1
  if 'loss' in hisDict:
    plt.subplot(210+pltIndex)
    loss = hisDict['loss']
    epochs = range(1, len(loss) + 1)  
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training and validation loss')
  if 'val_loss' in hisDict:
    val_loss = hisDict['val_loss']
    epochs = range(1, len(val_loss) + 1)  
    plt.plot(epochs, val_loss, 'b', label='Validation loss')  
  if 'acc' in hisDict:
    plt.subplot(210+pltIndex)
    pltIndex +=1
    acc = hisDict['acc']
    epochs = range(1, len(acc) + 1)  
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.title('Training and validation accuracy')
  if 'val_acc' in hisDict:
    val_acc = hisDict['val_acc']
    epochs = range(1, len(val_acc) + 1)  
    plt.plot(epochs, val_acc, 'b', label='Validation acc')


  plt.legend()
  plt.show()