import torch


class BMSELoss(torch.nn.Module):

    def __init__(self):
        super(BMSELoss, self).__init__()
        self.w_l = [1, 2, 5, 10, 30]
        self.y_l = [0.283, 0.353, 0.424, 0.565, 1]
        p=np.loadtxt('mask.txt')
        self.nan_index=p.astype(int)

    def forward(self, x, y):
        w = y.clone()
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]
        w[self.nan_index==1]=0
        return torch.mean(w * ((y - x)** 2) )
