import copy
import os.path as osp
import subprocess

import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import nvidia_dlprof_pytorch_nvtx


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        if args['amp']:
                x = x.half()
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')
        if args['amp']:
            x_all = x_all.half()

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

def train(epoch, grad_scaler):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args['amp']):
          y = batch.y[:batch.batch_size]
          y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
          loss = F.cross_entropy(y_hat, y)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        #optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test():
    model.eval()
    with torch.cuda.amp.autocast(enabled=args['amp']):
        y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
        y = data.y.to(y_hat.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs

def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])

    return result

if __name__ == "__main__":
    # Uncomment to use wandb for graph visualzation
    # import wandb
    # wandb.init()

    # Set amp to true to use fp16
    args = {
    'num_layers': 3,
    'hidden_dim': 16,
    'dropout': 0.5,
    'lr': 0.005,
    'wd': 5*10**(-4),
    'epochs': 300,
    'GPUdevice': 3,
    'amp': False,
    'profile': False,
    }

    device = torch.device('cuda:' + str(args['GPUdevice']) if torch.cuda.is_available() else 'cpu')
    print(device)
    init_mem = int(get_gpu_memory_map().decode("utf-8").split('\n')[args['GPUdevice']])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)

    # Already send node features/labels to GPU for faster access during sampling:
    data = dataset[0].to(device, 'x', 'y')
    kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
    train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                num_neighbors=[25, 10], shuffle=True, **kwargs)

    subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                    num_neighbors=[-1], shuffle=False, **kwargs)

    # No need to maintain these features during evaluation:
    del subgraph_loader.data.x, subgraph_loader.data.y
    # Add global node index information.
    subgraph_loader.data.num_nodes = data.num_nodes
    subgraph_loader.data.n_id = torch.arange(data.num_nodes)

    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Start training
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args['amp'])
    if (args['profile']): # If profile true use DLprof to profile fun
        nvidia_dlprof_pytorch_nvtx.init()
        with torch.autograd.profiler.emit_nvtx():
            for epoch in range(1, 2):
                loss, acc = train(epoch, grad_scaler)
                print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
                train_acc, val_acc, test_acc = test()
                print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                    f'Test: {test_acc:.4f}')
    else:
        for epoch in range(1, 2):
            loss, acc = train(epoch, grad_scaler)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
            train_acc, val_acc, test_acc = test()
            print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                f'Test: {test_acc:.4f}')
    final_mem = int(get_gpu_memory_map().decode("utf-8").split('\n')[args['GPUdevice']])
    print("Memory used: ", final_mem - init_mem)