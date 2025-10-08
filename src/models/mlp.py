import torch
import torch.nn as nn


class clf_mlpv2(nn.Module):
    def __init__(self, num_classes=2, num_tokens=96):
        super(clf_mlpv2, self).__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.hidden = nn.Linear(num_tokens, 4 * num_tokens)
        self.head = nn.Linear(4 * num_tokens, num_outputs)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x -> (b, 96, 4, 4, 4, t)
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.hidden(x)
        x = self.head(x)
        return x


class clf_mlpv1(nn.Module):
    def __init__(self, num_classes=2, num_tokens=96):
        super(clf_mlpv1, self).__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.head = nn.Linear(num_tokens, num_outputs)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x -> (b, 96, 4, 4, 4, t)
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


# Contrastice loss code adapted from TCLR: Temporal Contrastive Learning for Video Representation
class emb_mlp(nn.Module):

    def __init__(
        self,
        final_embedding_size=128,
        num_tokens=196,
        use_normalization=True,
        n_local_frames=4,
    ):

        super(emb_mlp, self).__init__()

        self.final_embedding_size = final_embedding_size
        self.use_normalization = use_normalization
        self.fc1 = nn.Linear(num_tokens, self.final_embedding_size, bias=False)
        self.bn1 = nn.BatchNorm1d(self.final_embedding_size)
        self.temp_avg = nn.AdaptiveAvgPool1d(1)  #
        self.n_local_frames = n_local_frames

    def forward(self, x, type):
        # x -> b, 96, 4, 4, 4, t

        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C

        if type == "l":
            # Global Dense Representation, this will be used for IC and LL losses
            x = self.temp_avg(x.transpose(1, 2))
            x = x.flatten(1)
            x = nn.functional.normalize(self.bn1(self.fc1(x)), p=2, dim=1)
            return x

        elif type == "g":
            # Global Sparse Representation, this will be used for the IC loss
            gsr = self.temp_avg(x.transpose(1, 2))
            gsr = gsr.flatten(1)
            gsr = nn.functional.normalize(self.bn1(self.fc1(gsr)), p=2, dim=1)
            return gsr

        else:
            return None, None, None, None, None
