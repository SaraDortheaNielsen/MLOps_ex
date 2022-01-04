import argparse
import sys

import torch
import torch.nn.functional as F
from torch import nn, optim

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        images = train_set['images']
        images = images.view(images.shape[0], -1)
        labels = train_set['labels']

        print(images.shape)
        #tmp = train_set['images']
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        epochs = 30
        steps = 0
        train_losses, test_losses = [], []
        train_accuracy, test_accuracy = [], []
        #train_losses, test_losses = [], []
        for e in range(epochs):
            running_loss = 0
            #for images, labels in trainloader:
                
            optimizer.zero_grad()
                
            log_ps = model(images.float())

            # loss
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            #running_loss += loss.item()
            train_losses.append(running_loss)

            # accuray
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            #running_accuracy_train += accuracy.item()
            train_accuracy.append(accuracy.item())

            print("\n done \n")
        
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, test_set = mnist()
        images_test = test_set['images']
        images_test = images_test.view(images_test.shape[0], -1)
        labels_test = test_set['labels']

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    