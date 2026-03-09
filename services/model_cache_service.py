"""
Model Caching Service
Menyimpan dan memuat model yang sudah dilatih untuk menghindari retraining
"""

import os
import pickle
import time
from datetime import datetime
from config import MODEL_CACHE_DIR, MODEL_CACHE_TTL

# Buat direktori cache jika belum ada
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


def get_cache_path(cache_key):
    """Get cache file path for a specific cache key (e.g., 'short_20260130')"""
    return os.path.join(MODEL_CACHE_DIR, f"{cache_key}_model.pkl")


def get_cache_metadata_path(cache_key):
    """Get cache metadata file path for a specific cache key"""
    return os.path.join(MODEL_CACHE_DIR, f"{cache_key}_metadata.pkl")


def is_cache_valid(cache_key):
    """Check if cached model is still valid (not expired)
    
    Args:
        cache_key: Full cache key including horizon and date (e.g., 'short_20260130')
    """
    cache_path = get_cache_path(cache_key)
    metadata_path = get_cache_metadata_path(cache_key)
    
    if not os.path.exists(cache_path) or not os.path.exists(metadata_path):
        return False
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Validate metadata structure
        if 'timestamp' not in metadata:
            return False
        
        cache_age = time.time() - metadata['timestamp']
        is_valid = cache_age < MODEL_CACHE_TTL
        
        return is_valid
    except Exception as e:
        print(f"Error checking cache validity: {e}")
        return False


def load_cached_model(cache_key):
    """Load model from cache
    
    Args:
        cache_key: Full cache key including horizon and date (e.g., 'short_20260130')
    """
    cache_path = get_cache_path(cache_key)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded cached {cache_key} model")
        return model
    except Exception as e:
        print(f"Error loading cached model: {e}")
        return None


def save_model_to_cache(cache_key, model):
    """Save model to cache
    
    Args:
        cache_key: Full cache key including horizon and date (e.g., 'short_20260130')
        model: The trained model to cache
    """
    cache_path = get_cache_path(cache_key)
    metadata_path = get_cache_metadata_path(cache_key)
    
    try:
        # Save model
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'timestamp': time.time(),
            'cache_key': cache_key,
            'saved_at': datetime.now().isoformat()
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Cached {cache_key} model")
    except Exception as e:
        print(f"Error saving model to cache: {e}")


def clear_cache(cache_key=None, horizon_prefix=None):
    """Clear cache for specific cache key, horizon prefix, or all caches
    
    Args:
        cache_key: Full cache key to clear (e.g., 'short_20260130')
        horizon_prefix: Clear all caches for a horizon (e.g., 'short' clears short_*)
        If both None, clears all cache
    """
    if cache_key:
        # Clear specific cache key
        cache_path = get_cache_path(cache_key)
        metadata_path = get_cache_metadata_path(cache_key)
        
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            print(f"✓ Cleared cache for {cache_key}")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    elif horizon_prefix:
        # Clear all caches for a horizon prefix
        try:
            for file in os.listdir(MODEL_CACHE_DIR):
                if file.startswith(f"{horizon_prefix}_"):
                    file_path = os.path.join(MODEL_CACHE_DIR, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            print(f"✓ Cleared all cache for {horizon_prefix}")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    else:
        # Clear all cache files
        try:
            for file in os.listdir(MODEL_CACHE_DIR):
                file_path = os.path.join(MODEL_CACHE_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("✓ Cleared all cache")
        except Exception as e:
            print(f"Error clearing all cache: {e}")
