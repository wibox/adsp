from tqdm import tqdm

import torch

import os

class UnetTrainer():
    def __init__(
        self,
        model,
        optimizer,
        loss,
        scaler,
        epochs,
        batch_size,
        train_loader,
        val_loader,
        device,
        checkpoint_path,
        checkpoint_filename
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.scaler = scaler
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_filename = checkpoint_filename

    def _initialise(self):
        torch.cuda.empty_cache()
        #self.device = torch.device("cuda:1")
        self.model.to(self.device)

    def _compute_metrics(self, mask, predicted_mask):
        mask_predicted_mask = (torch.sigmoid(predicted_mask)>.5).float()

        num_pixels += torch.numel(mask_predicted_mask)
        TP += (mask_predicted_mask == mask).sum()
        FP += (mask_predicted_mask - mask).sum()
        FN += (mask - mask_predicted_mask).sum()

        IoUScore += TP / (TP + FP + FN)

        predicted_pixels = TP/num_pixels
        iou_score = IoUScore/len(self.val_loader)

        print(f" ===> Predicted pixels: {predicted_pixels} with accuracy: {predicted_pixels*100:2f}")
        print(f" ===> IoU Score: {iou_score}")
        return predicted_pixels, iou_score

    def _train(self):
        # train loop
        loop = tqdm(self.train_loader)
        loss_10_batches = 0
        loss_epoch = 0
        for idx, (image, mask) in enumerate(loop):
            # data to GPU
            image = image.to(self.device)
            mask = mask.to(self.device)
            print("MASK SHAPE",mask.shape)
            # with torch.cuda.amp.autocast():
            out = self.model(image)
            print("PREDICTION SHAPE", out.shape)
            # if pos_weight:
            #     loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(mask==0.).sum()/mask.sum())
            loss = self.loss(out, mask)
            loss_10_batches += loss
            loss_epoch += loss
            self._compute_metrics(mask = mask, predicted_mask=out)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # update tqdm loop
        if idx%self.batch_size==0:
            loop.set_postfix(loss_10_batches=loss_10_batches.item()/self.batch_size)

        print(f"==> training_loss: {loss_epoch/len(self.train_loader):2f}")

    def _eval(self):

        loss_epoch = 0

        self.model.eval()
        with torch.no_grad():
            for idx, (image, mask) in enumerate(self.val_loader):
                image = image.to(self.device)
                mask = mask.to(self.device)
                predicted_mask = self.model(image)

                loss_epoch += self.loss(predicted_mask, mask)
                self._compute_metrics(mask = mask, predicted_mask=predicted_mask)
        print(f" ===> Validation Loss: {loss_epoch/len(self.val_loader):2f}")

        self.model.train()

    def _save_checkpoint(self, state):
        if not os.path.exist(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        print(f"Saving model's checkpoint in {self.checkpoint_path}...")
        torch.save(state, os.path.join(self.checkpoint_path, self.checkpoint_filename))

    def _load_checkpoint(self, checkpoint):
        loaded_checkpoint = torch.load(checkpoint)
        print("Loading model's checkpoint...")
        self.model.load_state_dict(loaded_checkpoint['state_dict'])
        print("Loading optimizer's checkpoint...")
        self.optimizer.load_state_dict(loaded_checkpoint['optimizer'])

    def train(self):
        self._initialise()
        for epoch in range(self.epochs):
            print(f"Training epoch: {epoch}")
            self._train()
            print("Computing IoUScore on validation dataset")
            self._eval()
