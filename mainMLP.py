import pandas as pd
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
import torch.optim as optim 
import copy 
import matplotlib.pyplot as plt 
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用するデバイス: {device}")

    df = pd.read_csv("train.tsv", sep = "\t")

    def create_data_list(df):
        data_list_x = []
        i = 0
        data_list_y = []
        for index, row in df.iterrows():
            text = row["sentence"]
            token = text.split(" ")
            label = row["label"]
            feature = Counter(token)
            data_list_x.append(feature)
            data_list_y.append(label)
        return data_list_x, data_list_y

    data_list_x, data_list_y = create_data_list(df)

    train_x, test_x, train_y, test_y = train_test_split(
        data_list_x, data_list_y, 
        test_size=0.2, 
        random_state=42
    )

    vectorizer = DictVectorizer(sparse=False)
    train_x_vec = vectorizer.fit_transform(train_x)
    test_x_vec = vectorizer.transform(test_x)

    train_x_tensor = torch.tensor(train_x_vec, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)

    test_x_tensor = torch.tensor(test_x_vec, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.long)

    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)

    train_loader  = DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=True
        )

    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
    test_loader  = DataLoader(
        test_dataset, 
        batch_size=1000,
        shuffle=True
        ) 

    class SimpleMLP(nn.Module):
        def __init__(self, input_dim = train_x_vec.shape[1], hidden_dim = 100):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
        def forward(self, x):
            out = self.net(x)
            #out = torch.clamp(out, min=-10, max=10)
            return out
        
    model = SimpleMLP()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
            
    optimizer = optim.Adam(model.parameters())

    epoch_num = 1000

    loss_train_list = []
    loss_test_list = []
    acu_list = []
    loss_min = float("inf")
    count_ear = 0

    for epoch in range(epoch_num):
        
        model.train()
        loss_train = 0
        for train_x, train_y in train_loader:
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            optimizer.zero_grad()
            y = model(train_x)
            loss = criterion(y, train_y)
            loss_train += loss.item()
            loss.backward()
            optimizer.step()
        ave_loss_train = loss_train/len(train_loader)
        loss_train_list.append(ave_loss_train)
        
        model.eval()
        
        with torch.no_grad():
            loss_test = 0
            correct = 0
            for  test_x, test_y in test_loader:
                test_x = test_x.to(device)
                test_y = test_y.to(device)
                y = model(test_x)
                loss = criterion(y, test_y)
                loss_test += loss.item()
                #accuracy
                _, predicted = torch.max(y, 1)
                correct += (predicted == test_y).sum().item()
            
            acu = correct/len(test_dataset)
            acu_list.append(acu)
            
            ave_loss_test = loss_test/len(test_loader)
            loss_test_list.append(ave_loss_test)
            print(f"epoch_num:{epoch}, train_loss:{ave_loss_train}, test_loss:{ave_loss_test}")

            if loss_min >= ave_loss_test:
                loss_min = ave_loss_test
                count_ear = 0
                acu_score = acu
            
            else:
                count_ear += 1
                
            if count_ear == 5:
                break
                
    print(f"accuracy score: ", {acu_score})
    # --- (学習ループが完了した後) ---
    print(f"Best model saved! Loss: {loss_min:.4f}")
    # 損失をグラフ化
    plt.figure(figsize=(10, 5))
    plt.plot(loss_train_list, label="Train Loss")
    plt.plot(loss_test_list, label="Validation Loss")
    plt.plot(acu_list, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("learning")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_graph.png")
    
if __name__ == "__main__":
    main()
