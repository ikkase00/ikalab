import torch
import torch.nn as nn

no = False
yes = True

# conditional number generator GAN model
class conditional_num_g(nn.Module):
    def __init__(self, imshape: list, classes:int, ):
        super(conditional_num_g, self).__init__()

        self.noif = 140 # initializing the noise vector
        self.classes = classes # identify number of classes
        self.emb = nn.Embedding(self.classes, self.classes) # creating word embedding layer
        self.base = 64 # base input channel
        self.imshape = imshape # identify image dimensions

        # creating the model
        self.ggc_ = nn.Sequential(
            *self.mache((self.classes + self.noif), self.base * 2, no),
            *self.mache(self.base * 2, self.base * 4),
            *self.mache(self.base * 4, self.base * 8),
            *self.mache(self.base * 8, self.base * 16),
            nn.Linear(self.base * 16, int(torch.prod(torch.tensor(self.imshape[:])))),
            nn.Tanh(),
        )

    # helper function to construct internal layers
    def mache(self, ic, oc, bnq=True, ):
        ll = [nn.Linear(ic, oc)]
        if bnq:
            ll.append(nn.BatchNorm1d(oc))
        ll.append(nn.LeakyReLU(2e-1, inplace=yes))

        return ll

    # forward pass
    def forward(self, noiz, labels, ):
        emblem = torch.cat((self.emb(labels), noiz), -1)
        out = self.ggc_(emblem)
        outt = out.view(out.size(0), *self.imshape)

        return outt

    # get noise
    def get(self, batch_size, label: int or str):
        """ 
        :param label: choose a number(0 <= int <= 9) / random generate(any string)
        :return: noise, according labels
        """
        assert (type(label) == str) or (0 <= label <= 9), 'pass a string or a int > 0 and < 9'
        if type(label) == str:
            la = torch.randint(0, 10, (batch_size, ), device = self.devi)
        else:
            la = torch.tensor([label for _ in range(batch_size)], device = self.devi)

        return torch.randn((batch_size, self.noif), device = self.devi), la

# vanilla + group CNN network
class hybrid_disc(nn.Module):
    def __init__(self, basek):
        """
        This model takes MNIST numbers (or number images generated from the generator from the same project) images tensor with shape batch_size, 1, 64, 64 as input.
        :param basek: base quantity of convolution filters
        """
        super(hybrid_disc, self).__init__()

        self.basek = basek

        self.driv = nn.Sequential(
            self.mache(1, self.basek // 2, 4, 2, 1, 1),
            self.mache(self.basek // 2, self.basek, 4, 2, 1, 8),
            self.mache(self.basek, self.basek * (2 ** 1), 4, 2, 1, 8),
            nn.Flatten(),
            nn.Linear(8192, 2048),
            nn.Dropout(.3),
            nn.PReLU(),
            nn.Linear(2048, 10),
            nn.Dropout(.3),
        )

    # helper function to make layers
    @staticmethod
    def mache(ic, oc, ks, st, pd, grp):
        """
        :param oc: note that this block generates GROUPED convolution, make sure the kernel quantity suits your target
        :param grp: channels per group
        """
        return nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ic, oc, ks, st, pd, groups = grp),
            nn.Conv2d(oc, oc, 1, 1, 0),
            nn.BatchNorm2d(oc, affine = yes),
        )

    # forward pass
    def forward(self, dat):
        return self.driv(dat).softmax(dim = -1)
