# coreComponents/tokenization.py

import pandas as pd
from typing import List, Dict, Optional
from shapely.geometry import Point
import h3


class TrajectoryTokenizer:
    """
    Simple tokenizer for H3 geohashes.
    Handles conversion between GPS points and H3 tokens.
    """
    
    def __init__(self, resolution=8):
        self.resolution = resolution
        self.stoi = {
            "<pad>": 0,
            "<unk>": 1,
            "<end>": 2,
            '<|endoftext|>': 3,
            'summarize:': 4
        }
        self.itos = {v: k for k, v in self.stoi.items()}
        self.next_id = len(self.stoi)
    
    # -----------------------------------------------------------------
    # Core Tokenization/Detokenization Methods
    # -----------------------------------------------------------------
    
    @staticmethod
    def gps_string_to_h3(gps_string: str, resolution: int) -> str:
        """
        Convert GPS string "lat1 lon1,lat2 lon2,..." to H3 token string.
        
        Args:
            gps_string: Raw GPS points string
            resolution: H3 resolution
            
        Returns:
            Space-separated H3 tokens
        """
        if not gps_string or not isinstance(gps_string, str):
            return ""
        
        # Clean the string
        gps_string = gps_string.strip('"\' ')
        if not gps_string:
            return ""
        
        tokens = []
        for point_str in gps_string.split(','):
            point_str = point_str.strip()
            if not point_str:
                continue
            
            parts = point_str.split()
            if len(parts) >= 2:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    # Convert to H3
                    h3_token = h3.latlng_to_cell(lat, lon, resolution)
                    tokens.append(h3_token)
                except:
                    continue
        
        return " ".join(tokens)
    
    @staticmethod
    def h3_to_gps(h3_string: str, bert_imputer=None) -> List[tuple]:
        """
        Convert H3 token string to GPS coordinates.
        
        Args:
            h3_string: Space-separated H3 tokens
            bert_imputer: Optional BERTImputer for better accuracy
            
        Returns:
            List of (lat, lon) tuples
        """
        if not h3_string:
            return []
        
        tokens = h3_string.split()
        coordinates = []
        previous_point = None
        
        for token in tokens:
            if bert_imputer:
                # Use BERTImputer if available
                point = bert_imputer.token2point_cluster_centroid(token, previous_point)
                coordinates.append((point.y, point.x))  # lat, lon
                previous_point = point
            else:
                # Use standard H3 conversion
                try:
                    lat, lon = h3.cell_to_latlng(token)
                    coordinates.append((lat, lon))
                except:
                    coordinates.append((0.0, 0.0))
        
        return coordinates
    
    # -----------------------------------------------------------------
    # Vocabulary Management
    # -----------------------------------------------------------------
    
    def build_vocab(self, h3_strings: List[str]):
        """
        Build vocabulary from list of H3 token strings.
        
        Args:
            h3_strings: List of space-separated H3 token strings
        """
        all_tokens = []
        for h3_string in h3_strings:
            if not h3_string:
                continue
            tokens = h3_string.split()
            all_tokens.extend(tokens)
        
        unique_tokens = sorted(set(all_tokens))
        for token in unique_tokens:
            if token not in self.stoi:
                self.stoi[token] = self.next_id
                self.itos[self.next_id] = token
                self.next_id += 1
    
    def encode(self, h3_string: str) -> List[int]:
        """
        Encode H3 token string to integer IDs.
        
        Args:
            h3_string: Space-separated H3 tokens
            
        Returns:
            List of integer token IDs
        """
        tokens = h3_string.split()
        ids = []
        for token in tokens:
            ids.append(self.stoi.get(token, self.stoi["<unk>"]))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode integer IDs back to H3 token string.
        
        Args:
            ids: List of integer token IDs
            
        Returns:
            Space-separated H3 tokens
        """
        tokens = []
        for token_id in ids:
            token = self.itos.get(token_id, "<unk>")
            tokens.append(token)
        return " ".join(tokens)
    
    # -----------------------------------------------------------------
    # CSV Processing
    # -----------------------------------------------------------------
    
    @staticmethod
    def process_csv(
        df: pd.DataFrame,
        columns_to_tokenize: List[str],
        resolution: int = 8
    ) -> pd.DataFrame:
        """
        Process CSV by converting GPS strings in specified columns to H3 tokens.
        
        Args:
            df: Input DataFrame
            columns_to_tokenize: List of column names to convert
            resolution: H3 resolution
            
        Returns:
            Processed DataFrame with H3 token strings
        """
        df_processed = df.copy()
        
        for col in columns_to_tokenize:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].apply(
                    lambda x: TrajectoryTokenizer.gps_string_to_h3(x, resolution)
                    if pd.notna(x) else ""
                )
        
        return df_processed
    
    # -----------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------
    
    def pad_token_id(self):
        return self.stoi["<pad>"]
    
    def eos_token_id(self):
        return self.stoi["<end>"]
    
    def get_vocab_size(self):
        return len(self.stoi)
    
    def is_spatial(self, result) -> bool:
        """
        Check if result contains spatial tokens.
        
        Args:
            result: Model output to check
            
        Returns:
            True if result is spatial
        """
        if not isinstance(result, list) or len(result) == 0:
            return False
        
        # Check if it's a list of H3 tokens (strings)
        if all(isinstance(token, str) and h3.is_valid_cell(token) for token in result):
            return True
        
        # Check if it's a list of integer IDs
        if all(isinstance(token, int) for token in result):
            return True
        
        return False