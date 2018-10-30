import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch.optim as optim
import pickle
from scipy.io import loadmat
import numpy as np
from data2dnn import convert_feats
from keras.preprocessing import sequence

# HYPERPARAMETERS
MODEL_NUM = 'E37_0.001'        # model number, for saving model & scores
HIDDEN_SIZE = 50      # number of hidden nodes
START_EPOCH = 0
NUM_EPOCHS = 22         # number of passes over the whole dataset
XAVIER = True           # if True, use Xavier weight initialization
DROPOUT = 0.00          # if nonzero, apply dropout with specified prob.
LEARNING_RATE = 0.001   # speed of convergence
OPTIM = 'sgd'           # optimizer to use; 'adam' or 'sgd'
MOMENTUM = 0.9          # accelerates gradient vectors in the right direction
BATCH_NORM = True       # if True, apply batch normalization
INPUT_SIZE = 90*11      # number of CQCC features +/- 5 context frames
NUM_CLASSES =  2        # number of output classes; 0=genuine, 1=spoof
DATA_DIR = '/group/project/'
SAVE = False
LOAD = False
SCORE_DEV = True
SCORE_EVAL = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

print("Training model Py-DNN{}".format(MODEL_NUM))
print("Loading data...")
# Load TRAINING data
train_fp = np.memmap(DATA_DIR+'extracted_feats/train-feats-window',
            dtype='float32', mode='r', shape=(938016,990))
tensor_train_x = torch.from_numpy(train_fp)

with open(DATA_DIR+'targets/train-targets.pickle', 'rb') as f:
    train_targets = pickle.load(f)
tensor_train_y = torch.LongTensor(train_targets)
trainset = utils.TensorDataset(tensor_train_x, tensor_train_y)
trainloader = utils.DataLoader(trainset, batch_size=2000, shuffle=True)

# Load DEV data
dev_fp = np.memmap(DATA_DIR+'extracted_feats/dev-feats-window',
            dtype='float32', mode='r', shape=(607779,990))
tensor_dev_x = torch.from_numpy(dev_fp)
with open(DATA_DIR+'targets/dev-targets.pickle', 'rb') as f:
    dev_targets = pickle.load(f)
tensor_dev_y = torch.LongTensor(dev_targets)
devset = utils.TensorDataset(tensor_dev_x, tensor_dev_y)
devloader = utils.DataLoader(trainset, batch_size=2000)
print("Done!")


## DEFINE NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc5 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc6 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc7 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        if BATCH_NORM:
            self.bn1 = nn.BatchNorm1d(HIDDEN_SIZE)
            self.bn2 = nn.BatchNorm1d(HIDDEN_SIZE)
            self.bn3 = nn.BatchNorm1d(HIDDEN_SIZE)
            self.bn4 = nn.BatchNorm1d(HIDDEN_SIZE)
            self.bn5 = nn.BatchNorm1d(HIDDEN_SIZE)
            self.bn6 = nn.BatchNorm1d(HIDDEN_SIZE)
            self.bn7 = nn.BatchNorm1d(NUM_CLASSES)
        if DROPOUT > 0:
        	self.dropout = nn.Dropout(DROPOUT)

        if XAVIER: #initialize weights
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)

    def forward(self, x):
    	'''
    	Do a forward pass through the network.
    	'''
        if BATCH_NORM:
            if DROPOUT > 0:
                x = self.dropout(F.sigmoid(self.bn1(self.fc1(x))))
                x = self.dropout(F.sigmoid(self.bn2(self.fc2(x))))
                x = self.dropout(F.sigmoid(self.bn3(self.fc3(x))))
                x = self.dropout(F.sigmoid(self.bn4(self.fc4(x))))
                x = self.dropout(F.sigmoid(self.bn5(self.fc5(x))))
                x = self.dropout(F.sigmoid(self.bn6(self.fc6(x))))
                x = F.sigmoid(self.bn7(self.fc7(x)))
            else:
            	 x = F.sigmoid(self.bn1(self.fc1(x)))
                x = F.sigmoid(self.bn2(self.fc2(x)))
                x = F.sigmoid(self.bn3(self.fc3(x)))
                x = F.sigmoid(self.bn4(self.fc4(x)))
                x = F.sigmoid(self.bn5(self.fc5(x)))
                x = F.sigmoid(self.bn6(self.fc6(x)))
                x = F.sigmoid(self.bn7(self.fc7(x)))
        elif DROPOUT > 0:
            x = self.dropout(F.sigmoid(self.fc1(x)))
            x = self.dropout(F.sigmoid(self.fc2(x)))
            x = self.dropout(F.sigmoid(self.fc3(x)))
            x = self.dropout(F.sigmoid(self.fc4(x)))
            x = self.dropout(F.sigmoid(self.fc5(x)))
            x = self.dropout(F.sigmoid(self.fc6(x)))
            x = F.sigmoid(self.fc7(x))
        else:
            x = F.sigmoid(self.fc1(x)) #pass through 1st hidden layer
            x = F.sigmoid(self.fc2(x)) #pass through 2nd hidden layer
            x = F.sigmoid(self.fc3(x)) #pass through 3rd hidden layer
            x = F.sigmoid(self.fc4(x))
            x = F.sigmoid(self.fc5(x))
            x = F.sigmoid(self.fc6(x))
             x = F.sigmoid(self.fc7(x))
        return x

def train(net, start, epochs):
	'''
	Train the network for the specified number of epochs, starting from
	a particular epoch.
	'''
    ## TRAIN
    dev_losses = []
    train_losses = []
    for epoch in range(start, NUM_EPOCHS):
        print("Training model Py-DNN{}".format(MODEL_NUM))
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data # get inputs
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # clear existing gradients

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49: # print every 50 minibatches
                print('[Epoch {}, Batch {}] loss: {:.4f}'.format(epoch+1, i+1,
                      running_loss/50))
                train_losses.append(running_loss/50)
                running_loss = 0.0
        compute_dev_loss(net, epoch, dev_losses)
        if SCORE_DEV:
            score_dev(net, MODEL_NUM, epoch)
    if SCORE_EVAL:
        score_eval(net, MODEL_NUM, epoch)
    if SAVE:
        torch.save(net, './models/py-dnn{}.pt'.format(MODEL_NUM))

    print("Done! Finished training model Py-DNN{}".format(MODEL_NUM))


def compute_dev_loss(net, epoch, dev_losses):
	'''
	Compute loss on validation set after each epoch.
	'''
	print("Computing loss on dev set...")
	net.eval()
	with torch.no_grad():
	    dev_loss_per_epoch = []
	    for i, data in enumerate(devloader, 0):
	        inputs, labels = data
	        inputs, labels = inputs.to(device), labels.to(device)
	        outputs = net(inputs)
	        dev_loss = loss_fn(outputs, labels)
	        dev_loss_per_epoch.append(dev_loss)
	    epoch_dev_loss = sum(dev_loss_per_epoch) / len(dev_loss_per_epoch)
	    dev_losses.append(epoch_dev_loss)
	    print("Epoch {} Dev Loss: {:.3f}".format(epoch+1, epoch_dev_loss))


def score_dev(net, model_num, epoch):
	'''
	Use the neural network to score each sample in the validation set and
	create a file of those scores. Score after each epoch.
	'''
    print("Writing DEV set scores file...")
    feats = loadmat(DATA_DIR+'extracted_feats/dev-feats')
    feat_cell = feats['devFeatureCell']

    with open(DATA_DIR+'scores/py-dnn{}-ep{}_dev_scores.csv'.format(MODEL_NUM, epoch+1),
              'a') as f:
        with torch.no_grad():
            net.eval()
            CONTEXT = 5
            for utt in feat_cell:
                utt = utt[0].transpose()
                utt = np.concatenate((np.zeros((CONTEXT,90)), utt))
                utt = np.concatenate((utt, np.zeros((CONTEXT,90))))

                utt_context = []
                for j, frame in enumerate(utt):
                    if j < CONTEXT:
                        continue
                    elif j >= len(utt)-CONTEXT:
                        break
                    else:
                    	frame_context = utt[j-CONTEXT]
                        for k in range(1,CONTEXT+1):
                            frame_context = np.concatenate((frame_context,
                                                            utt[j-CONTEXT+k]))
                        for k in range(1,CONTEXT+1):
                            frame_context = np.concatenate((frame_context,
                                                           utt[j+k]))
                        utt_context.append(frame_context)
                utt_context = np.array(utt_context)
                tensx = torch.from_numpy(utt_context)
                tensx = tensx.type(torch.FloatTensor)
                outputs = net(tensx)
                gen_scores = [o[1] for o in outputs]
                utt_score = sum(gen_scores) / len(gen_scores)
                f.write(str(float(utt_score)) + '\n')
        print("Done!")


def score_eval(net, model_num, epoch):
	'''
	Use the neural network to score each sample in the evaluation set and
	create a file of those scores. Score after each epoch.
	'''
	print("Writing EVAL set scores file...")
    CONTEXT = 5
    feat_cells = []
    feats = loadmat(DATA_DIR+'extracted_feats/eval-feats-A')
    feat_cells.append(feats['evalfeatcellA'])
    feats = loadmat(DATA_DIR+'extracted_feats/eval-feats-B')
    feat_cells.append(feats['evalfeatcellB'])

    with open(DATA_DIR+'scores/py-dnn{}-ep{}_eval_scores.csv'.format(MODEL_NUM,
              epoch+1), 'a') as f:
        with torch.no_grad():
            net.eval()
            for feat_cell in feat_cells:
                for utt in feat_cell:
                    utt = utt[0].transpose()
                    utt = np.concatenate((np.zeros((CONTEXT,90)), utt))
                    utt = np.concatenate((utt, np.zeros((CONTEXT,90))))

                    utt_context = []
                    for j, frame in enumerate(utt):
                        if j < CONTEXT:
                            continue
                        elif j >= len(utt)-CONTEXT:
                            break
                        else:
                            frame_context = utt[j-CONTEXT]
                            for k in range(1,CONTEXT+1):
                                frame_context = np.concatenate((frame_context,
                                                                utt[j-CONTEXT+k]))
                            for k in range(1,CONTEXT+1):
                                frame_context = np.concatenate((frame_context,
                                                               utt[j+k]))
                        utt_context.append(frame_context)
                    utt_context = np.array(utt_context)
                    tensx = torch.from_numpy(utt_context)
                    tensx = tensx.type(torch.FloatTensor)
                    outputs = net(tensx)
                    gen_scores = [o[1] for o in outputs]
                    utt_score = sum(gen_scores) / len(gen_scores)
                    f.write(str(float(utt_score)) + '\n')
            print("Done!")


if __name__ == '__main__':
    if LOAD:
        net = torch.load('./models/py-dnn{}.pt'.format(MODEL_NUM))
    else:
        net = Net()

    ## DEFINE LOSS FUNCTION & OPTIMIZER
    loss_fn = nn.CrossEntropyLoss()
    if OPTIM == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE,
                              momentum=MOMENTUM)
    elif OPTIM == 'adam':
    	optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, amsgrad=True)

    train(net, start=START_EPOCH, epochs=NUM_EPOCHS)