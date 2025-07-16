import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
from .model import Tacotron2
from .data_processor import DatasetProcessor

class VoiceDataset(Dataset):
    def __init__(self, data_dir: str, speaker_id: str):
        self.data_dir = Path(data_dir)
        self.speaker_id = speaker_id
        
        # Load metadata
        with open(self.data_dir / speaker_id / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load features
        features = torch.load(self.data_dir / speaker_id / 'features.pt')
        self.mel_spectrograms = features['mel_spectrograms']
        self.text_sequences = features['text_sequences']
    
    def __len__(self):
        return len(self.mel_spectrograms)
    
    def __getitem__(self, idx):
        text_seq = torch.LongTensor(self.text_sequences[idx])
        mel_spec = self.mel_spectrograms[idx]
        
        # Create gate target (1 at the end, 0 elsewhere)
        gate_target = torch.zeros(mel_spec.size(1))
        gate_target[-1] = 1.0
        
        return text_seq, mel_spec, gate_target

def collate_fn(batch):
    """Collate function for DataLoader."""
    text_lengths = torch.LongTensor([len(x[0]) for x in batch])
    mel_lengths = torch.LongTensor([x[1].size(1) for x in batch])
    
    max_text_len = torch.max(text_lengths).item()
    max_mel_len = torch.max(mel_lengths).item()
    
    # Pad sequences
    text_padded = torch.LongTensor(len(batch), max_text_len).zero_()
    mel_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_mel_len).zero_()
    gate_padded = torch.FloatTensor(len(batch), max_mel_len).zero_()
    
    for i, (text, mel, gate) in enumerate(batch):
        text_padded[i, :text.size(0)] = text
        mel_padded[i, :, :mel.size(1)] = mel
        gate_padded[i, :gate.size(0)] = gate
    
    return text_padded, text_lengths, mel_padded, gate_padded, mel_lengths

class VoiceTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Tacotron2(self.config['model']).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate']
        )
        
        self.criterion = Tacotron2Loss()
        
        # Create directories
        self.checkpoint_dir = Path(self.config['paths']['checkpoints'])
        self.log_dir = Path(self.config['paths']['logs'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_dataloaders(self, data_dir: str, speaker_id: str, 
                           val_split: float = 0.1) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation dataloaders."""
        dataset = VoiceDataset(data_dir, speaker_id)
        
        # Split dataset
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = [item.to(self.device) for item in batch]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch[:5])  # text, text_lens, mel, max_len, mel_lens
            loss = self.criterion(outputs, batch[2:4])  # mel_target, gate_target
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [item.to(self.device) for item in batch]
                outputs = self.model(batch[:5])
                loss = self.criterion(outputs, batch[2:4])
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch: int, loss: float, speaker_id: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'speaker_id': speaker_id
        }
        
        checkpoint_path = self.checkpoint_dir / f'{speaker_id}_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Saved checkpoint: {checkpoint_path}')
    
    def train(self, data_dir: str, speaker_id: str):
        """Main training loop."""
        self.logger.info(f'Starting training for speaker: {speaker_id}')
        
        # Prepare data
        train_loader, val_loader = self.prepare_dataloaders(data_dir, speaker_id)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['max_epochs']):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            if epoch % self.config['training']['validation_interval'] == 0:
                val_loss = self.validate(val_loader)
                self.logger.info(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(epoch, val_loss, speaker_id)
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    self.logger.info('Early stopping triggered')
                    break
            
            # Save checkpoint
            if epoch % self.config['training']['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch, train_loss, speaker_id)
        
        self.logger.info('Training completed')

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, model_output, targets):
        mel_target, gate_target = targets
        mel_out, mel_out_postnet, gate_out, _ = model_output
        
        mel_loss = self.mse_loss(mel_out, mel_target) + \
                   self.mse_loss(mel_out_postnet, mel_target)
        gate_loss = self.bce_loss(gate_out, gate_target)
        
        return mel_loss + gate_loss

class VoiceOnboarder:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.data_processor = DatasetProcessor(config_path)
        self.trainer = VoiceTrainer(config_path)
    
    def onboard_new_voice(self, voice_data_dir: str, speaker_id: str, 
                         output_dir: str) -> str:
        """Complete pipeline to onboard a new voice."""
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Starting voice onboarding for: {speaker_id}')
        
        try:
            # 1. Process voice data
            self.logger.info('Processing voice data...')
            speaker_data = self.data_processor.process_voice_data(
                voice_data_dir, speaker_id
            )
            
            # 2. Save processed data
            self.data_processor.save_processed_data(speaker_data, output_dir)
            
            # 3. Train model
            self.logger.info('Training voice model...')
            self.trainer.train(output_dir, speaker_id)
            
            model_path = self.trainer.checkpoint_dir / f'{speaker_id}_best.pt'
            self.logger.info(f'Voice onboarding completed. Model saved at: {model_path}')
            
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f'Error during voice onboarding: {e}')
            raise e
