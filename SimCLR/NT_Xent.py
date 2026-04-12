import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        """
        temperature: The scaling parameter tau. The paper found 0.5 works best 
                     when training to convergence (e.g. > 300 epochs)[cite: 864].
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        """
        z_i: The projected vectors from the first set of augmented views.
        z_j: The projected vectors from the second set of augmented views.
        Both should have shape (batch_size, projection_dim).
        """
        batch_size = z_i.size(0)

        # 1. L2 Normalize the vectors along the projection dimension [cite: 98]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 2. Concatenate all views to shape (2 * batch_size, projection_dim)
        representations = torch.cat([z_i, z_j], dim=0)

        # 3. Calculate cosine similarity matrix for all pairs
        # Resulting shape: (2 * batch_size, 2 * batch_size)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # 4. Scale by the temperature parameter [cite: 104]
        similarity_matrix = similarity_matrix / self.temperature

        # 5. Create labels to identify the positive pairs.
        # For an image at index 'k' in z_i, its positive pair is at index 'k + batch_size' in z_j.
        # And vice-versa.
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0)
        labels = labels.to(similarity_matrix.device)

        # 6. Mask out the diagonals (an image's similarity to itself should not be in the loss)
        # We create a mask of negative infinity for the diagonals
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(similarity_matrix.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # 7. Calculate the Cross Entropy Loss
        # The network tries to predict the correct 'label' (the positive pair's index) 
        # out of all the possible indices in the similarity matrix.
        loss = self.criterion(similarity_matrix, labels)
        
        # Return the average loss per positive pair [cite: 105, 109]
        return loss / (2 * batch_size)