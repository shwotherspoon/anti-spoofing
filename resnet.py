import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch.optim as optim
import pickle
from scipy.io import loadmat
import numpy as np
from data2dnn import convert_feats
import argparse


# HYPERPARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('model_num')
parser.add_argument('hidden', type=int)
parser.add_argument('lr', type=float)
args = parser.parse_args()
MODEL_NUM = args.model_num 			# model number, for saving model & scores
HIDDEN_SIZE = args.hidden 			# number of hidden nodes
LEARNING_RATE = args.lr 			# speed of convergence
print(MODEL_NUM, type(MODEL_NUM))
print(HIDDEN_SIZE, type(HIDDEN_SIZE))
print(LEARNING_RATE, type(LEARNING_RATE))
START_EPOCH = 0
NUM_EPOCHS = 25         # number of passes over the whole dataset
XAVIER = True           # if True, use Xavier weight initialization
DROPOUT = 0.00          # if nonzero, apply dropout with specified prob.
OPTIM = 'adam'           # optimizer to use; 'adam' or 'sgd'
MOMENTUM = 0.9          # accelerates gradient vectors in the right direction
INPUT_SIZE = 90*11      # number of CQCC features +/- 5 context frames
NUM_CLASSES =  2        # number of output classes; 0=genuine, 1=spoof
DATA_DIR = '/group/project/'
SAVE = False
LOAD = False
SCORE_DEV = True
SCORE_EVAL = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

print("Training model ResNet{}".format(MODEL_NUM))
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
        # Pre-blocks
        self.fc0 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        # Block 1
        self.bn1a = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc1a = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.bn1b = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc1b = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # Block 2
        self.bn2a = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc2a = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.bn2b = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc2b = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # Block 3
        self.bn3a = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc3a = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.bn3b = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc3b = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # Block 4
        self.bn4a = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc4a = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.bn4b = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc4b = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # # Block 5
        # self.bn5a = nn.BatchNorm1d(HIDDEN_SIZE)
        # self.fc5a = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # self.bn5b = nn.BatchNorm1d(HIDDEN_SIZE)
        # self.fc5b = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # # Block 6
        # self.bn6a = nn.BatchNorm1d(HIDDEN_SIZE)
        # self.fc6a = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # self.bn6b = nn.BatchNorm1d(HIDDEN_SIZE)
        # self.fc6b = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # Post-blocks
        self.bn7 = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc7 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

        if XAVIER: #initialize weights
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)

    def forward(self, x):
    	'''
    	Do a forward pass through the network.
    	'''
    	# Pre-blocks
        x = self.fc0(x)
        # Block 1
        residual = x.clone()
        x = F.relu(self.bn1a(x))
        x = F.relu(self.bn1b(self.fc1a(x)))
        x = self.fc1b(x)
        x += residual
        # Block 2
        residual = x.clone()
        x = F.relu(self.bn2a(x))
        x = F.relu(self.bn2b(self.fc2a(x)))
        x = self.fc2b(x)
        x += residual
        # Block 3
        residual = x.clone()
        x = F.relu(self.bn3a(x))
        x = F.relu(self.bn3b(self.fc3a(x)))
        x = self.fc3b(x)
        x += residual
        # Block 4
        residual = x.clone()
        x = F.relu(self.bn4a(x))
        x = F.relu(self.bn4b(self.fc4a(x)))
        x = self.fc4b(x)
        x += residual
        # # Block 5
        # residual = x
        # x = F.relu(self.bn5a(x))
        # x = F.relu(self.bn5b(self.fc5a(x)))
        # x = self.fc5b(x)
        # x = x + residual
        # # Block 6
        # residual = x
        # x = F.relu(self.bn6a(x))
        # x = F.relu(self.bn6b(self.fc6a(x)))
        # x = self.fc6b(x)
        # x = x + residual
        # Post-blocks
        x = F.relu(self.bn7(x))
         x = self.fc7(x)
        x = F.softmax(x, dim=1)
        return x


def train(net, start, epochs):
	'''
	Train the network for the specified number of epochs, starting from
	a particular epoch.
	'''
    dev_losses = []
    train_losses = []
    for epoch in range(start, NUM_EPOCHS):
        print("Training model ResNet{}".format(MODEL_NUM))
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
            score_dev(net, epoch)
        if SCORE_EVAL:
            score_eval(net, epoch)
    if SAVE:
        torch.save(net, './models/ResNet{}.pt'.format(MODEL_NUM))

    print("Done! Finished training model ResNet{}".format(MODEL_NUM))


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


def score_dev(net, epoch):
	'''
	Use the neural network to score each sample in the validation set and
	create a file of those scores. Score after each epoch.
	'''
	print("Writing DEV set scores file...")
    feats = loadmat(DATA_DIR+'extracted_feats/dev-feats')
    feat_cell = feats['devFeatureCell']

    with open(DATA_DIR+'scores/ResNet{}-ep{}_dev_scores.csv'.format(MODEL_NUM, epoch+1),
              'a') as f:
        with torch.no_grad():
            net.eval()
            CONTEXT = 5
            scores = []
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


def score_eval(net, epoch):
	'''
	Use the neural network to score each sample in the evaluation set and
	create a file of those scores. Score after each epoch.
	'''
    print("Writing EVAL set scores file...")
    CONTEXT = 5
    feats_A = loadmat(DATA_DIR+'extracted_feats/eval-feats-A')
    feat_cell_A = feats_A['evalfeatcellA']
    feats_B = loadmat(DATA_DIR+'extracted_feats/eval-feats-B')
    feat_cell_B = feats_B['evalfeatcellB']
    feat_cells = [feat_cell_A, feat_cell_B]

    with open(DATA_DIR+'scores/ResNet{}-ep{}_eval_scores.csv'.format(MODEL_NUM, epoch+1),
              'a') as f:
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
        net = torch.load('./models/ResNet{}.pt'.format(MODEL_NUM))
    else:
        net = Net()

    ## DEFINE LOSS FUNCTION & OPTIMIZER
    loss_fn = nn.CrossEntropyLoss()
    if OPTIM == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE,
                              momentum=MOMENTUM)
    elif OPTIM == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, amsgrad=True)

    train(net, START_EPOCH, NUM_EPOCHS)